"""Crown-owned settings.

``surya`` is treated as an unmodified pip dependency, so any *new*
configuration knobs that crown needs live here rather than in
``surya.settings``. Knobs that already exist in surya (e.g.
``SURYA_INFERENCE_PARALLEL``, ``LLAMA_CPP_NGL``) are still read from
``surya.settings.settings``; this module only adds crown-specific overrides
and the new tuning surface introduced for batch-processing parallelism.

The settings read the same ``local.env`` file surya uses (via
``dotenv.find_dotenv``) so a single env file configures both packages.
"""

from __future__ import annotations

from typing import Optional

from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class CrownSettings(BaseSettings):
    # ------------------------------------------------------------------
    # Client-side concurrency
    # ------------------------------------------------------------------
    # Decoupled from SURYA_INFERENCE_PARALLEL (server decode slots).
    # Server slots cost KV-cache memory; client workers just keep the HTTP
    # queue full so continuous batching stays saturated. None = backend
    # default:
    #   llamacpp -> surya SURYA_INFERENCE_PARALLEL
    #   vllm    -> the server's --max-num-seqs (saturate continuous batching)
    # Set explicitly when pointing at a shared external server
    # (SURYA_INFERENCE_URL) whose capacity you don't control.
    SURYA_INFERENCE_MAX_WORKERS: Optional[int] = None

    # ------------------------------------------------------------------
    # llama.cpp prompt-processing / CPU tuning
    # ------------------------------------------------------------------
    # Defaults are chosen to be safe on all platforms. Enable per-host via
    # env vars. These are only consumed by CrownLlamaCppBackend.start().
    #
    # -b: prompt-processing batch size. Larger helps the heavy vision
    # prefill. 2048 is a good default for surya-2's image inputs.
    LLAMA_CPP_BATCH: int = 2048
    # -ub: micro-batch (ubatch) size for prompt eval. Match -b by default.
    LLAMA_CPP_UBATCH: int = 512
    # -t: CPU threads for the non-offloaded parts of the graph. None =
    # max(1, cpu_count // 2). Set explicitly to pin a specific count.
    LLAMA_CPP_THREADS: Optional[int] = None
    # --flash-attn on: large throughput win on builds with flash-attention
    # kernels. Safe on current llama.cpp builds for the surya GGUF; if a
    # build lacks FA kernels it errors at startup, which the existing
    # health-check + log capture surfaces clearly.
    LLAMA_CPP_FLASH_ATTN: bool = True
    # --no-mmap: load the whole model into RAM instead of mmap-ing. Faster
    # first-token latency after load but raises resident RAM by the model
    # size. Default off.
    LLAMA_CPP_NO_MMAP: bool = False
    # --direct-io: bypass page cache for model file reads. Helps when the
    # model is on fast local disk. Default off.
    LLAMA_CPP_DIRECT_IO: bool = False
    # --numa <distribute|isolate|numactl>: NUMA strategy. None = off.
    # Only enable on multi-socket NUMA boxes; on single-socket hosts it can
    # reduce throughput.
    LLAMA_CPP_NUMA: Optional[str] = None
    # --cpu-range "0-15": pin worker threads to a CPU set. None = off.
    LLAMA_CPP_CPU_RANGE: Optional[str] = None
    # --cpu-strict 1: strict CPU affinity (only meaningful with --cpu-range).
    LLAMA_CPP_CPU_STRICT: bool = False
    OLLAMA_URL: Optional[str] = None
    OLLAMA_URL_LAYOUT: Optional[str] = None
    DEBUG_FOLDER: Optional[str] = None
    OLLAMA_SURYA_MODEL: Optional[str] = None
    OLLAMA_GLM_MODEL: Optional[str] = None
    TRIM_LARGE_IMAGES_LEFT_SIDE: bool = True
    LARGE_IMAGES_HOR_THRESHOLD: int = 12000

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


crown_settings = CrownSettings()