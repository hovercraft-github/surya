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


logger = get_logger()


class CrownVllmBackend(VllmBackend):
    """Crown-owned subclass of :class:`VllmBackend`."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()

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
            max_workers=settings.SURYA_INFERENCE_PARALLEL,
            request_logprobs_default=settings.SURYA_INFERENCE_LOGPROBS,
        )


class CrownLlamaCppBackend(LlamaCppBackend):
    """Crown-owned subclass of :class:`LlamaCppBackend`."""

    name = "llamacpp"

    def start(self):  # type: ignore[override]
        return super().start()

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
            max_workers=settings.SURYA_INFERENCE_PARALLEL,
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


class CrownSuryaInferenceManager(SuryaInferenceManager):
    """Crown-owned subclass of :class:`SuryaInferenceManager`.

    Construction is identical to the parent. Override methods here when crown
    needs to add logging, instrumentation, retry, or any other custom
    behavior; the rest of the codebase keeps using this type.
    """

    def __init__(self, method: Optional[str] = None, lazy: bool = True):
        super().__init__(method=method, lazy=True)
        self.backend: Backend = _build_backend(self.method)
        if not lazy:
            self.backend.start()

    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        super().stop()

    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        return super().generate(batch)


__all__ = [
    "CrownLlamaCppBackend",
    "CrownSuryaInferenceManager",
    "CrownVllmBackend",
]
