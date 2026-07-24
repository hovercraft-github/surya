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

import copy
import json
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from openai import OpenAI

from surya.inference import SuryaInferenceManager, _autodetect_backend
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


def get_hostname(url):
    if not url.startswith(('http://', 'https://', '//')):
        url = '//' + url
    return urlsplit(url).hostname

def get_port(url):
    if not url.startswith(('http://', 'https://', '//')):
        url = '//' + url
    return urlsplit(url).port

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


class CrownOllamaBackend(Backend):
    """Crown-owned backend that talks to an external Ollama server.

    Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so
    this backend is a thin wrapper around :func:`crown.openai_client.chat_completions_batch`.
    Unlike vllm/llamacpp it never spawns a server: the Ollama instance is
    assumed to be already running at the URL configured by
    ``OLLAMA_URL_LAYOUT``. The default model is taken from
    ``OLLAMA_SURYA_MODEL``. A dummy API key is sent because Ollama ignores it
    but the OpenAI client requires one to be set.
    """

    name = "ollama"

    def __init__(self) -> None:
        self.handle: Optional[ServerHandle] = None
        self._client: Optional[OpenAI] = None

    def start(self) -> ServerHandle:
        if self.handle is not None:
            return self.handle
        base_url = crown_settings.OLLAMA_URL_LAYOUT
        if not base_url:
            raise ValueError(
                "OLLAMA_URL_LAYOUT is not set; cannot start the ollama backend."
            )
        url_path = urlsplit(base_url).path
        if not url_path.startswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        model_name = crown_settings.OLLAMA_SURYA_MODEL or ""
        if not model_name:
            raise ValueError(
                "OLLAMA_SURYA_MODEL is not set; cannot start the ollama backend."
            )
        self.handle = ServerHandle(
            base_url=base_url,
            model_name=model_name,
            spawned_by_us=False,
        )
        self._client = OpenAI(api_key="ollama", base_url=self.handle.base_url)
        return self.handle

    def stop(self) -> None:
        # We never spawn the Ollama server, so there is nothing to stop.
        self.handle = None
        self._client = None

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        if self.handle is None or self._client is None:
            self.start()
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=settings.SURYA_INFERENCE_TIMEOUT_SECONDS,
            max_workers=_ollama_max_workers(),
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )


def _ollama_max_workers() -> int:
    """Client-side concurrency for the ollama backend.

    Defaults to the same tuning surface as the other backends
    (``SURYA_INFERENCE_MAX_WORKERS``), falling back to surya's
    ``SURYA_INFERENCE_PARALLEL`` when unset.
    """
    return crown_settings.SURYA_INFERENCE_MAX_WORKERS or settings.SURYA_INFERENCE_PARALLEL


def _build_backend(method: str) -> Backend:
    method = method.lower()
    if method == "vllm":
        return CrownVllmBackend()
    if method == "llamacpp":
        return CrownLlamaCppBackend()
    if method == "ollama":
        return CrownOllamaBackend()
    raise ValueError(
        f"Unknown inference backend {method!r}. Supported: 'vllm', 'llamacpp', 'ollama'."
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
    _batch_started_at: Optional[float] = None
    _last_batch_duration: Optional[float] = None

    # Booking service state.
    _booked_requests: int = 0

    # Batcher state: pending inputs and completed outputs keyed by call_id.
    # Each value is (timestamp, items) where timestamp is time.monotonic()
    # for expiration. Guarded by _batcher_lock.
    _input_batches: Dict[str, Tuple[float, List[BatchInputItem]]] = {}
    _output_batches: Dict[str, Tuple[float, List[BatchOutputItem]]] = {}
    _batcher_lock = threading.Lock()
    _batcher_thread: Optional[threading.Thread] = None
    _batcher_stop = threading.Event()
    # Polling interval (seconds) for generate() to probe output_batches and
    # for the batcher thread to wake up and check input_batches.
    _batcher_poll_interval: float = 0.05
    # Entries older than this (seconds, monotonic) are expired from the
    # input/output maps to bound memory in pathological cases.
    _batcher_entry_ttl: float = 600.0

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        # Enforce single-instance invariant. Acquiring the lock here also
        # serializes concurrent construction attempts.
        with CrownSuryaInferenceManager._instance_lock:
            super().__init__(method="llamacpp", lazy=True)
            self.method = method or _autodetect_backend()
            if CrownSuryaInferenceManager._instance is None:
                CrownSuryaInferenceManager._instance = self
            # This trick is against fastapi and other forks: TODO: verify on Windows and MacOS. If this fails, we may need to use a more robust singleton pattern.
            elif CrownSuryaInferenceManager._instance is not self:
                self.backend: Backend = _build_backend(self.method)
        # Start the background batcher thread that drains input_batches and
        # fills output_batches. Daemonized so it never blocks process exit.
        self._ensure_batcher_thread()

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
        with self._batcher_lock:
            is_running = self._batcher_thread is not None and self._batcher_thread.is_alive()
            if is_running:
                return
            if not hasattr(self, "backend"):
                self.backend = _build_backend(self.method)
            if self.method in ["vllm", "llamacpp"]:
                super().start()
            else:
                settings.SURYA_INFERENCE_URL = crown_settings.OLLAMA_URL_LAYOUT
                settings.SURYA_INFERENCE_AUTOSTART = False
            self._ensure_batcher_thread()

    def stop(self) -> None:
        # Signal the background batcher thread to exit, then join it.
        self._batcher_stop.set()
        thread = self._batcher_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=self._default_batch_estimate)
        self._batcher_thread = None
        self._batcher_stop.clear()
        super().stop()

    # -- batcher thread ---------------------------------------------------

    def _ensure_batcher_thread(self) -> None:
        """Start the background send_batch loop if it isn't running."""
        if self._batcher_thread is not None and self._batcher_thread.is_alive():
            return
        self._batcher_stop.clear()
        thread = threading.Thread(
            target=self._send_batch_loop,
            name="CrownSuryaInferenceManager-batcher",
            daemon=True,
        )
        self._batcher_thread = thread
        thread.start()

    def _send_batch_loop(self) -> None:
        """Background loop: drain input_batches into super().generate()."""
        while not self._batcher_stop.is_set():
            try:
                self.send_batch()
            except Exception:
                logger.exception("send_batch failed; will retry")
            # Wait for the poll interval or until stopped.
            self._batcher_stop.wait(self._batcher_poll_interval)

    def send_batch(self) -> None:
        """Combine pending input_batches into one planar batch and dispatch it.

        Runs on the background batcher thread. When the number of pending
        call_ids in :attr:`_input_batches` reaches ``self._booked_requests``,
        all pending items are flattened into a single batch, tagged with their
        originating ``call_id`` in ``metadata``, passed to ``super().generate``,
        and the results are split back out by ``call_id`` into
        :attr:`_output_batches`.
        """
        start_time = time.monotonic()
        # Snapshot the pending inputs under the lock.
        with self._batcher_lock:
            self._batch_started_at = start_time
            self._expire_entries()
            if not self._input_batches:
                return
            # Trigger when we have at least _booked_requests pending calls,
            # or when there is only one call pending and no further bookings
            # are expected (best-effort: avoids stalling a lone caller).
            pending_count = len(self._input_batches)
            if self._booked_requests > 0 and pending_count < self._booked_requests:
                return
            # Take ownership of the pending inputs.
            pending = self._input_batches
            self._input_batches = {}

        # Build a planar batch preserving order, and remember which call_id
        # each item belongs to so we can split results back out.
        planar_batch: List[BatchInputItem] = []
        for call_id, (_ts, items) in pending.items():
            for item in items:
                # Tag the item with its call_id in metadata so the backend
                # round-trips it back to us on the output side.
                tagged = copy.copy(item)
                tagged.metadata = dict(item.metadata)
                tagged.metadata["call_id"] = call_id
                planar_batch.append(tagged)

        if not planar_batch:
            return

        results = super().generate(planar_batch)

        # Split results back out by call_id.
        per_call: Dict[str, List[BatchOutputItem]] = {}
        for out_item in results:
            cid = out_item.metadata["call_id"]
            if cid in per_call:
                per_call[cid].append(out_item)
            else:
                per_call[cid] = [out_item]

        now = time.monotonic()
        with self._batcher_lock:
            for cid, items in per_call.items():
                if items:
                    self._output_batches[cid] = (now, items)
            self._last_batch_duration = time.monotonic() - start_time
            self._batch_started_at = None

    def _expire_entries(self) -> None:
        """Drop expired input/output entries. Caller holds _batcher_lock."""
        cutoff = time.monotonic() - self._batcher_entry_ttl
        for store in (self._input_batches, self._output_batches):
            for key in [k for k, (ts, _) in store.items() if ts < cutoff]:
                del store[key]

    # -- batch generation -------------------------------------------------

    def _estimate_retry_after(self) -> float:
        """Estimate how many seconds until the in-flight batch completes."""
        with self._batcher_lock:
            if self._last_batch_duration is not None:
                # Use the most recent observed duration as the estimate.
                elapsed = time.monotonic() - (self._batch_started_at or time.monotonic() + 5.0)
                remaining = self._last_batch_duration - elapsed
                return max(remaining, self._default_batch_estimate)
        # No history yet: fall back to the configured default estimate.
        return self._default_batch_estimate

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        """Enqueue a batch and block until the batcher thread produces results.

        Each call gets a unique ``call_id`` (uuid4). The batch is stored in
        :attr:`_input_batches` keyed by ``call_id`` with a monotonic timestamp
        for expiration, and every :class:`BatchInputItem` is tagged with the
        same ``call_id`` in its ``metadata`` dict. The method then polls
        :attr:`_output_batches` for the matching ``call_id`` and returns the
        results.
        """
        call_id = str(uuid.uuid4())
        now = time.monotonic()
        with self._batcher_lock:
            self._input_batches[call_id] = (now, list(batch))

        # Block until the batcher thread publishes results for this call_id.
        poll = self._batcher_poll_interval
        while not self._batcher_stop.is_set():
            with self._batcher_lock:
                entry = self._output_batches.pop(call_id, None)
            if entry is not None:
                _ts, results = entry
                return results
            time.sleep(poll)
        # Shutting down: return whatever we can, or raise.
        raise RuntimeError("CrownSuryaInferenceManager is shutting down")

    @asynccontextmanager
    async def booking(self):
        """Async context manager to limit the number of concurrent API requests."""
        # max_workers = crown_settings.SURYA_INFERENCE_MAX_WORKERS or 1
        limit = crown_settings.SURYA_INFERENCE_MAX_WORKERS or settings.SURYA_INFERENCE_PARALLEL * 2

        if self._booked_requests >= limit:
            estimate = self._estimate_retry_after()
            raise BatchBusyError(retry_after=estimate)

        with self._batcher_lock:
            self._booked_requests += 1
        try:
            yield
        finally:
            with self._batcher_lock:
                self._booked_requests -= 1


def get_backend_host(manager: "SuryaInferenceManager") -> str:
    """Return the hostname of the backend server, if known."""
    if manager.backend is None:
        return "localhost"
    if isinstance(manager.backend, CrownOllamaBackend):
        return get_hostname(manager.backend.handle.base_url) if manager.backend.handle else "localhost"
    if isinstance(manager.backend, (CrownVllmBackend, CrownLlamaCppBackend)):
        return get_hostname(manager.backend.handle.base_url) if manager.backend.handle else "localhost"
    return "localhost"


def get_backend_port(manager: "SuryaInferenceManager") -> Optional[int]:
    """Return the port of the backend server, if known."""
    if manager.backend is None:
        return None
    if isinstance(manager.backend, CrownOllamaBackend):
        return get_port(manager.backend.handle.base_url) if manager.backend.handle else None
    if isinstance(manager.backend, (CrownVllmBackend, CrownLlamaCppBackend)):
        return get_port(manager.backend.handle.base_url) if manager.backend.handle else None
    return None

__all__ = [
    "BatchBusyError",
    "CrownLlamaCppBackend",
    "CrownOllamaBackend",
    "CrownSuryaInferenceManager",
    "CrownVllmBackend",
]
