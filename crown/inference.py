"""Crown subclasses of Surya's inference manager and backends.

These are intentionally thin pass-through subclasses that mirror the parent
contracts. They exist so that `surya_api.py` (and any other crown code) can
import a stable, crown-owned type — meaning future customizations to the
inference lifecycle can happen in this module without touching the upstream
`surya` package or the call sites that depend on it.

The bodies of the methods are deliberately minimal. Override what you need;
leave the rest to `super()`.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from surya.inference import SuryaInferenceManager
from surya.inference.backends.llamacpp import LlamaCppBackend, _resolve_llama_server_binary, _download_gguf_files, _health_url, _openai_url
from surya.inference.backends.vllm import VllmBackend, _gpu_settings, _resolve_docker_binary
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.spawn import (
    SpawnHandle,
    SpawnError,
    attach_or_spawn,
)
from surya.logging import get_logger
from surya.settings import settings


from crown.openai_client import chat_completions_batch
from crown.settings import crown_settings


logger = get_logger()


def _llamacpp_max_workers() -> int:
    """Client-side concurrency for the llama.cpp backend.

    Decoupled from the server's ``--parallel`` slots: client workers only
    keep the HTTP queue full, while slots cost KV-cache memory. Default to
    the server slot count (surya's SURYA_INFERENCE_PARALLEL) so the queue
    stays at least as full as the server can drain.
    """
    return crown_settings.SURYA_INFERENCE_MAX_WORKERS or settings.SURYA_INFERENCE_PARALLEL


def _vllm_max_workers(server_capacity: int) -> int:
    """Client-side concurrency for the vllm backend.

    Defaults to the server's ``--max-num-seqs`` so the HTTP queue saturates
    the continuous-batching capacity. Override via
    ``SURYA_INFERENCE_MAX_WORKERS`` (e.g. for a shared external server).
    """
    return crown_settings.SURYA_INFERENCE_MAX_WORKERS or server_capacity


class CrownVllmBackend(VllmBackend):
    """Crown-owned subclass of :class:`VllmBackend`."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        # Continuous-batching capacity of the spawned server, set in
        # start() and consumed by generate() as the client-concurrency
        # default so the HTTP queue keeps the server full.
        self._max_num_seqs: int = 0

    def start(self):  # type: ignore[override]
        if self.handle is not None:
            return self.handle

        # If user pinned an external server, attach without spawning docker.
        if settings.SURYA_INFERENCE_URL:
            spawned = attach_or_spawn(
                backend=self.name,
                expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
                spawn_fn=lambda port: SpawnHandle(
                    pid=None, cleanup_id="", cleanup_kind="docker"
                ),
                health_url_for=_health_url,
                openai_url_for=_openai_url,
                startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
            )
            self.handle = ServerHandle(
                base_url=spawned.base_url,
                model_name=spawned.model_name,
                spawned_by_us=spawned.spawned_by_us,
            )
            self._client = OpenAI(
                api_key=settings.VLLM_API_KEY, base_url=self.handle.base_url
            )
            return self.handle

        docker = _resolve_docker_binary()
        max_batched_tokens, max_num_seqs = _gpu_settings(settings.VLLM_GPU_TYPE)
        # Remember the server's continuous-batching capacity so generate()
        # can size the client worker pool to keep it saturated.
        self._max_num_seqs = max_num_seqs

        def spawn_fn(port: int) -> SpawnHandle:
            container_name = f"surya-vllm-{port}"
            hf_cache = os.path.expanduser(settings.DOCKER_HF_CACHE_PATH)
            cmd = [
                docker,
                "run",
                # "--rm",
                "-d",
                "--name",
                container_name,
                "--runtime",
                "nvidia",
                "--gpus",
                f"device={settings.VLLM_GPUS}",
                "-v",
                f"{hf_cache}:/root/.cache/huggingface",
                "-p",
                f"{port}:8000",
                "--ipc=host",
                settings.VLLM_DOCKER_IMAGE,
                "--model",
                settings.SURYA_MODEL_CHECKPOINT,
                "--no-enforce-eager",
                "--max-num-seqs",
                str(max_num_seqs),
                "--dtype",
                settings.VLLM_DTYPE,
                "--max-model-len",
                str(settings.VLLM_MAX_MODEL_LEN),
                "--max-num-batched-tokens",
                str(max_batched_tokens),
                "--gpu-memory-utilization",
                str(settings.VLLM_GPU_MEMORY_UTILIZATION),
                "--enable-prefix-caching",
                "--mm-processor-kwargs",
                json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
                "--served-model-name",
                settings.SURYA_MODEL_CHECKPOINT,
            ]
            if settings.VLLM_ENABLE_MTP:
                spec_config = json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": settings.VLLM_MTP_TOKENS,
                    }
                )
                cmd.extend(["--speculative-config", spec_config])
            for extra in (settings.VLLM_EXTRA_ARGS or "").split():
                cmd.append(extra)
            logger.info(f"Spawning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise SpawnError(f"docker run failed: {result.stderr or result.stdout}")
            return SpawnHandle(
                pid=None, cleanup_id=container_name, cleanup_kind="docker"
            )

        spawned = attach_or_spawn(
            backend=self.name,
            expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
            spawn_fn=spawn_fn,
            health_url_for=_health_url,
            openai_url_for=_openai_url,
            startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
        )
        self.handle = ServerHandle(
            base_url=spawned.base_url,
            model_name=spawned.model_name,
            spawned_by_us=spawned.spawned_by_us,
        )
        self._client = OpenAI(
            api_key=settings.VLLM_API_KEY,
            base_url=self.handle.base_url,
        )
        return self.handle

    def stop(self) -> None:
        super().stop()

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.handle is None or self._client is None:
            self.start()
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
            max_workers=_vllm_max_workers(max(1, self._max_num_seqs)),
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )


class CrownLlamaCppBackend(LlamaCppBackend):
    """Crown-owned subclass of :class:`LlamaCppBackend`.

    Overrides ``start()`` to spawn llama-server with crown's tuned
    prompt-processing flags (flash-attn, batch/ubatch sizing, CPU threads,
    optional NUMA/CPU pinning) gated by :mod:`crown.settings`. The parent's
    ``--parallel`` / ``--ctx-size`` scaling logic is preserved.
    """

    name = "llamacpp"

    def start(self):  # type: ignore[override]
        if self.handle is not None:
            return self.handle

        # If user pinned an external server, attach without spawning.
        # No binary or GGUF download needed in that case.
        if settings.SURYA_INFERENCE_URL:
            spawned = attach_or_spawn(
                backend=self.name,
                expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
                spawn_fn=lambda port: SpawnHandle(
                    pid=None, cleanup_id="", cleanup_kind="process"
                ),  # never called
                health_url_for=_health_url,
                openai_url_for=_openai_url,
                startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
            )
            self.handle = ServerHandle(
                base_url=spawned.base_url,
                model_name=spawned.model_name,
                spawned_by_us=spawned.spawned_by_us,
            )
            self._client = OpenAI(api_key="EMPTY", base_url=self.handle.base_url)
            return self.handle

        binary = _resolve_llama_server_binary()

        # Pre-download GGUFs so the spawn doesn't race the download
        if (
            settings.SURYA_GGUF_LOCAL_MODEL_PATH
            and settings.SURYA_GGUF_LOCAL_MMPROJ_PATH
        ):
            model_path = settings.SURYA_GGUF_LOCAL_MODEL_PATH
            mmproj_path = settings.SURYA_GGUF_LOCAL_MMPROJ_PATH
        else:
            model_path, mmproj_path = _download_gguf_files()

        # Total KV-cache budget. llama-server divides --ctx-size across
        # --parallel slots, so a too-small total silently truncates outputs
        # once each slot's share fills. Scale with parallel by default;
        # SURYA_INFERENCE_CTX_SIZE overrides to a fixed value if set.
        parallel = settings.SURYA_INFERENCE_PARALLEL
        per_slot = settings.SURYA_INFERENCE_CTX_PER_SLOT
        ctx_size = settings.SURYA_INFERENCE_CTX_SIZE
        if ctx_size is None:
            ctx_size = max(16384, parallel * per_slot)
        effective_per_slot = ctx_size // max(parallel, 1)
        logger.info(
            f"llama-server ctx-size={ctx_size} "
            f"(~{effective_per_slot}/slot × {parallel} parallel slots)"
        )
        if effective_per_slot < per_slot:
            logger.warning(
                f"per-slot ctx ({effective_per_slot}) is below recommended "
                f"{per_slot}; outputs may truncate. Raise "
                f"SURYA_INFERENCE_CTX_SIZE or SURYA_INFERENCE_CTX_PER_SLOT, "
                f"or lower SURYA_INFERENCE_PARALLEL."
            )

        # CPU threads for the non-offloaded parts of the graph.
        threads = crown_settings.LLAMA_CPP_THREADS
        if threads is None:
            threads = max(1, (os.cpu_count() or 1) // 2)

        def spawn_fn(port: int) -> SpawnHandle:
            cmd = [
                binary,
                "-m",
                model_path,
                "--mmproj",
                mmproj_path,
                "-b",
                str(crown_settings.LLAMA_CPP_BATCH),
                "-ub",
                str(crown_settings.LLAMA_CPP_UBATCH),
                "-t",
                str(threads),
                "-ngl",
                str(settings.LLAMA_CPP_NGL),
                "--host",
                settings.SURYA_INFERENCE_HOST,
                "--port",
                str(port),
                "--parallel",
                str(parallel),
                "--ctx-size",
                str(ctx_size),
                "--no-mmproj-offload" if settings.LLAMA_CPP_NO_MMPROJ_OFFLOAD else "",
                "--alias",
                settings.SURYA_MODEL_CHECKPOINT,
                "--jinja",
            ]
            cmd = [c for c in cmd if c]
            # Prompt-processing / CPU tuning (crown-owned, safe defaults).
            if crown_settings.LLAMA_CPP_FLASH_ATTN:
                cmd += ["--flash-attn", "on"]
            if crown_settings.LLAMA_CPP_NO_MMAP:
                cmd.append("--no-mmap")
            if crown_settings.LLAMA_CPP_DIRECT_IO:
                cmd.append("--direct-io")
            if crown_settings.LLAMA_CPP_NUMA:
                cmd += ["--numa", crown_settings.LLAMA_CPP_NUMA]
            if crown_settings.LLAMA_CPP_CPU_RANGE:
                cmd += ["--cpu-range", crown_settings.LLAMA_CPP_CPU_RANGE]
                if crown_settings.LLAMA_CPP_CPU_STRICT:
                    cmd += ["--cpu-strict", "1"]
            for extra in (settings.LLAMA_CPP_EXTRA_ARGS or "").split():
                cmd.append(extra)
            logger.info(f"Spawning: {' '.join(cmd)}")
            log_path = Path("~/.cache/datalab/surya/llamacpp_server.log").expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "ab")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return SpawnHandle(
                pid=proc.pid, cleanup_id=str(proc.pid), cleanup_kind="process"
            )

        spawned = attach_or_spawn(
            backend=self.name,
            expected_model_name=settings.SURYA_MODEL_CHECKPOINT,
            spawn_fn=spawn_fn,
            health_url_for=_health_url,
            openai_url_for=_openai_url,
            startup_timeout=settings.SURYA_INFERENCE_STARTUP_TIMEOUT,
        )
        self.handle = ServerHandle(
            base_url=spawned.base_url,
            model_name=spawned.model_name,
            spawned_by_us=spawned.spawned_by_us,
        )
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=self.handle.base_url,
        )
        return self.handle

    def stop(self) -> None:
        super().stop()

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.handle is None or self._client is None:
            self.start()
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
            max_workers=_llamacpp_max_workers(),
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )


def _build_backend(method: str) -> Backend:
    method = method.lower()
    if method == "vllm":
        return CrownVllmBackend()
    if method == "llamacpp":
        return CrownLlamaCppBackend()
    raise ValueError(
        f"Unknown inference backend {method!r}. Supported: 'vllm', 'llamacpp'."
    )


class BatchBusyError(Exception):
    """Raised when a new batch is scheduled while a previous one is still running.

    Carries an estimated ``retry_after`` value (in seconds) that the REST API
    should return to the client via the ``Retry-After`` HTTP header on a 429
    response.
    """

    def __init__(self, retry_after: float, message: Optional[str] = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            message
            or f"Another batch is already running; retry after {retry_after:.1f}s."
        )


class CrownSuryaInferenceManager(SuryaInferenceManager):
    """Crown-owned subclass of :class:`SuryaInferenceManager`.

    Construction is identical to the parent. Override methods here when crown
    needs to add logging, instrumentation, retry, or any other custom
    behavior; the rest of the codebase keeps using this type.

    This manager is a singleton: only one instance may exist at a time within
    a process. It also serializes batch generation: a single batch may run at
    any given moment. Attempting to start a new batch while a previous one is
    still in flight raises :class:`BatchBusyError` carrying an estimated
    ``Retry-After`` value (in seconds).
    """

    # Singleton bookkeeping.
    _instance: Optional["CrownSuryaInferenceManager"] = None
    _instance_lock = threading.Lock()

    # Heuristic: how long we expect a batch to take, used as the fallback
    # Retry-After estimate when no prior batch has completed yet.
    _default_batch_estimate: float = 30.0
    # Batch serialization state.
    _batch_lock = threading.Lock()
    _batch_running = False
    _batch_started_at: Optional[float] = None
    _last_batch_duration: Optional[float] = None

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        # Enforce single-instance invariant. Acquiring the lock here also
        # serializes concurrent construction attempts.
        with CrownSuryaInferenceManager._instance_lock:
            super().__init__(method=method, lazy=True)
            if CrownSuryaInferenceManager._instance is None:
                CrownSuryaInferenceManager._instance = self
            # This trick is against fastapi and other forks: TODO: verify on Windows and MacOS. If this fails, we may need to use a more robust singleton pattern.
            elif CrownSuryaInferenceManager._instance is not self:
                self.backend: Backend = _build_backend(self.method)

    # -- singleton access -------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CrownSuryaInferenceManager":
        """Return the single live instance, raising if none exists yet."""
        with cls._instance_lock:
            if cls._instance is None:
                raise RuntimeError(
                    "No CrownSuryaInferenceManager instance has been created yet."
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Clear the singleton slot. Intended for tests / teardown only."""
        with cls._instance_lock:
            cls._instance = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if not hasattr(self, "backend"):
            self.backend = _build_backend(self.method)
        super().start()

    def stop(self) -> None:
        super().stop()

    # -- batch generation -------------------------------------------------

    def _estimate_retry_after(self) -> float:
        """Estimate how many seconds until the in-flight batch completes."""
        if self._last_batch_duration is not None:
            # Use the most recent observed duration as the estimate.
            elapsed = time.monotonic() - (self._batch_started_at or time.monotonic())
            remaining = self._last_batch_duration - elapsed
            return max(remaining, 1.0)
        # No history yet: fall back to the configured default estimate.
        return self._default_batch_estimate

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        # Only one batch at a time. Try to acquire the batch lock without
        # blocking so we can immediately report a Retry-After estimate to the
        # caller instead of queueing behind the running batch.
        if not self._batch_lock.acquire(blocking=False):
            raise BatchBusyError(retry_after=self._estimate_retry_after())
        try:
            self._batch_running = True
            self._batch_started_at = time.monotonic()
            start = time.monotonic()
            result = super().generate(batch)
            self._last_batch_duration = time.monotonic() - start
            return result
        finally:
            self._batch_running = False
            self._batch_started_at = None
            self._batch_lock.release()


__all__ = [
    "BatchBusyError",
    "CrownLlamaCppBackend",
    "CrownSuryaInferenceManager",
    "CrownVllmBackend",
]
