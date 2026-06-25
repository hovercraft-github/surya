#!/home/al/local/AI-tools/surya-api/.venv/bin/python
# -*- coding: utf-8 -*-

"""Parallel-request stress test for the surya-api HTTP endpoints.

This is a *client-side* black-box test: it does **not** import `surya` or
`crown`. It talks to a running `surya_api` server over HTTP and measures how
the backend copes with many concurrent requests.

Design
------
* A single source image is reused for every request, but each request uploads
  it under a *unique filename* that encodes the test context:

      pass{PASS}_member{MEMBER}_par{PARALLEL}.png

  This lets the server logs / metrics attribute each in-flight request to a
  specific pass, member and parallelism level, which is the whole point of the
  exercise (see `plans/parallelism.md`).
* Requests inside one pass are issued concurrently with `asyncio.gather` using
  an `httpx.AsyncClient`. The number of simultaneous in-flight requests is the
  "parallel" value for that pass.
* Multiple passes can be configured; each pass may use a different parallelism
  level and/or endpoint, so you can ramp up load progressively.

Usage
-----
Run against a server started with::

    python surya_api.py --port 8522

Then::

    # default config (single pass, 4 parallel requests to /ocr/block/)
    python tests/api/test_parallel_requests.py

    # custom: ramp 1 -> 4 -> 8 -> 16 parallel requests, hit /ocr/full/
    python tests/api/test_parallel_requests.py \
        --endpoint /ocr/full/ \
        --passes 1 4 8 16 \
        --members 3 \
        --base-url http://127.0.0.1:8522

The script prints a per-pass summary (success/failure counts, latency stats,
wall-clock time, achieved throughput) and exits non-zero if any request failed
or if the success rate dropped below `--min-success-rate`.

It is intentionally *not* a pytest test (it needs a live server and is meant
for manual / CI-on-demand benchmarking), so it lives under `tests/api/` but is
run as a plain script.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

# Resolve the default source image relative to the repo root so the script can
# be invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = _REPO_ROOT / "static" / "images" / "excerpt.png"

# Endpoints exposed by surya_api.py (see surya_api.py:258 and surya_api.py:486).
KNOWN_ENDPOINTS = ("/ocr/full/", "/ocr/block/")


@dataclass
class RequestOutcome:
    """Result of a single parallel request."""

    member: int
    parallel: int
    status_code: int
    elapsed: float
    ok: bool
    error: str | None = None


@dataclass
class PassReport:
    """Aggregated results for one parallelism pass."""

    endpoint: str
    parallel: int
    members: int
    outcomes: list[RequestOutcome] = field(default_factory=list)

    @property
    def successes(self) -> list[RequestOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failures(self) -> list[RequestOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return len(self.successes) / len(self.outcomes)

    @property
    def latencies(self) -> list[float]:
        return [o.elapsed for o in self.successes]


def filename_for(pass_idx: int, member: int, slot: int, parallel: int) -> str:
    """Build the unique upload filename encoding the test context.

    The server logs `file.filename` (surya_api.py:311, surya_api.py:537), so
    this name propagates into the backend logs and lets us correlate each
    in-flight request with its pass/member/parallel slot.

    - ``pass``   : which parallelism level (pass) this request belongs to.
    - ``member`` : which repetition of the batch within the pass.
    - ``slot``   : which of the `parallel` concurrent slots this request occupies
                   (0 .. parallel-1), so individual in-flight requests are
                   distinguishable in the logs even within one member batch.
    - ``par``    : the concurrency level of this pass (echoed for convenience).
    """
    return f"pass{pass_idx}_member{member}_slot{slot}_par{parallel}.png"


async def fire_one(
    client: httpx.AsyncClient,
    endpoint: str,
    image_bytes: bytes,
    pass_idx: int,
    member: int,
    slot: int,
    parallel: int,
) -> RequestOutcome:
    """Issue a single OCR request and record its outcome."""
    filename = filename_for(pass_idx, member, slot, parallel)
    files = {"file": (filename, image_bytes, "image/png")}
    start = time.perf_counter()
    try:
        resp = await client.post(endpoint, files=files, timeout=None)
        elapsed = time.perf_counter() - start
        ok = resp.status_code == 200
        error = None if ok else f"HTTP {resp.status_code}: {resp.text[:200]}"
        return RequestOutcome(member, parallel, resp.status_code, elapsed, ok, error)
    except httpx.HTTPError as exc:
        elapsed = time.perf_counter() - start
        return RequestOutcome(member, parallel, 0, elapsed, False, str(exc))


async def run_pass(
    client: httpx.AsyncClient,
    endpoint: str,
    image_bytes: bytes,
    pass_idx: int,
    parallel: int,
    members: int,
) -> PassReport:
    """Run one parallelism pass.

    For each ``member`` repetition, fire ``parallel`` requests **concurrently**
    via ``asyncio.gather`` and wait for that batch to finish before starting the
    next member. So:

    - the *concurrency* level is ``parallel`` (the thing we are stress-testing);
    - ``members`` only increases the sample size by repeating the same
      concurrency level several times, sequentially.

    Total requests for the pass = ``parallel * members``.
    """
    report = PassReport(endpoint=endpoint, parallel=parallel, members=members)
    for member in range(members):
        tasks = [
            fire_one(client, endpoint, image_bytes, pass_idx, member, slot, parallel)
            for slot in range(parallel)
        ]
        report.outcomes.extend(await asyncio.gather(*tasks))
    return report


def fmt_stats(values: Iterable[float]) -> str:
    vals = list(values)
    if not vals:
        return "n/a"
    return (
        f"min={min(vals):.2f}s "
        f"median={statistics.median(vals):.2f}s "
        f"mean={statistics.mean(vals):.2f}s "
        f"max={max(vals):.2f}s"
    )


def print_report(report: PassReport, wall_time: float) -> None:
    total = len(report.outcomes)
    succ = len(report.successes)
    fail = len(report.failures)
    throughput = (succ / wall_time) if wall_time > 0 else 0.0
    print(
        f"\n=== Pass: endpoint={report.endpoint} parallel={report.parallel} "
        f"members={report.members} total={total} ==="
    )
    print(f"  wall time : {wall_time:.2f}s")
    print(f"  success   : {succ}/{total} ({report.success_rate * 100:.1f}%)")
    print(f"  failures  : {fail}")
    print(f"  latency   : {fmt_stats(report.latencies)}")
    print(f"  throughput: {throughput:.2f} req/s")
    if report.failures:
        # Show up to a few distinct failure reasons for triage.
        seen: dict[str, int] = {}
        for o in report.failures:
            key = o.error or f"status={o.status_code}"
            seen[key] = seen.get(key, 0) + 1
        print("  failure reasons:")
        for reason, count in seen.items():
            print(f"    [{count}] {reason}")


async def main_async(args: argparse.Namespace) -> int:
    image_path: Path = args.image
    if not image_path.is_file():
        print(f"ERROR: source image not found: {image_path}", file=sys.stderr)
        return 2
    image_bytes = image_path.read_bytes()

    base_url = args.base_url.rstrip("/")
    endpoint = args.endpoint
    if endpoint not in KNOWN_ENDPOINTS:
        print(f"ERROR: unknown endpoint {endpoint!r}", file=sys.stderr)
        return 2

    url = f"{base_url}{endpoint}"
    print(f"Target: {url}")
    print(f"Source image: {image_path} ({len(image_bytes)} bytes)")
    print(f"Passes (parallel levels): {args.passes}")
    print(f"Members per pass: {args.members}")

    limits = httpx.Limits(max_connections=max(args.passes) * args.members + 8)
    overall_ok = True
    # Set base_url on the client so the bare endpoint path ("/ocr/block/")
    # resolves to an absolute URL in fire_one().
    async with httpx.AsyncClient(base_url=base_url, limits=limits) as client:
        # Sanity check: server reachable?
        try:
            health = await client.get("/docs", timeout=10)
            if health.status_code != 200:
                print(
                    f"WARNING: /docs returned {health.status_code}; "
                    "server may not be healthy.",
                    file=sys.stderr,
                )
        except httpx.HTTPError as exc:
            print(f"ERROR: cannot reach server at {base_url}: {exc}", file=sys.stderr)
            return 3

        for pass_idx, parallel in enumerate(args.passes):
            start = time.perf_counter()
            report = await run_pass(
                client, endpoint, image_bytes, pass_idx, parallel, args.members
            )
            wall_time = time.perf_counter() - start
            print_report(report, wall_time)
            if report.success_rate < args.min_success_rate:
                overall_ok = False

    print("\nDone." if overall_ok else "\nDone with failures.")
    return 0 if overall_ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send parallel OCR requests to surya-api to stress-test "
        "backend parallelism. Reuses one image, renaming it per "
        "pass/member/parallel slot."
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8522",
        help="Base URL of the running surya_api server (default: %(default)s).",
    )
    p.add_argument(
        "--endpoint",
        default="/ocr/block/",
        choices=KNOWN_ENDPOINTS,
        help="API endpoint to hit (default: %(default)s).",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Source image file reused for every request (default: %(default)s).",
    )
    p.add_argument(
        "--passes",
        type=int,
        nargs="+",
        default=[4],
        help="Parallelism levels to test, one pass per value. Each pass fires "
        "parallel*members concurrent requests (default: 4).",
    )
    p.add_argument(
        "--members",
        type=int,
        default=1,
        help="Number of members (repetitions) per parallel slot within a pass. "
        "Total concurrent requests per pass = parallel * members (default: %(default)s).",
    )
    p.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help="Minimum fraction of successful requests for the run to be "
        "considered passing (default: %(default)s).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())