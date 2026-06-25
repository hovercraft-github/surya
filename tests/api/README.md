# surya-api parallel-request tests

These are **black-box HTTP tests** for the `surya_api` server. They do **not**
import `surya` or `crown` — they talk to a running server over HTTP, so they
belong to the future standalone `surya_api`/`crown` project, not to the `surya`
pip dependency.

## `test_parallel_requests.py`

Stress-tests the backend's parallel capabilities by firing many concurrent OCR
requests. A single source image is reused for every request, but each one is
uploaded under a unique filename encoding the test context:

```
pass{PASS}_member{MEMBER}_par{PARALLEL}.png
```

Because the server logs `file.filename` ([`surya_api.py`](../../surya_api.py)),
this name propagates into the backend logs and lets you correlate each in-flight
request with its pass / member / parallel slot — the whole point of the exercise
(see [`plans/parallelism.md`](../../plans/parallelism.md)).

### Prerequisites

Start the server in one terminal:

```bash
.venv/bin/python surya_api.py --port 8522
```

### Run

```bash
# default: one pass, 4 parallel requests to /ocr/block/
.venv/bin/python tests/api/test_parallel_requests.py

# ramp up parallelism: 1 -> 4 -> 8 -> 16, hit /ocr/full/
.venv/bin/python tests/api/test_parallel_requests.py \
    --endpoint /ocr/full/ --passes 1 4 8 16

# more samples per level (members) and a relaxed success threshold
.venv/bin/python tests/api/test_parallel_requests.py \
    --passes 4 8 16 --members 3 --min-success-rate 0.9
```

### Options

| option               | meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `--base-url`         | Base URL of the running server (default `http://127.0.0.1:8522`).       |
| `--endpoint`         | `/ocr/full/` or `/ocr/block/` (default `/ocr/block/`).                  |
| `--image`            | Source image reused for every request (default `static/images/excerpt.png`). |
| `--passes`           | Parallelism levels, one pass per value (default `4`).                   |
| `--members`          | Repetitions of the concurrent batch; total reqs per pass = `parallel * members` (default `1`). |
| `--min-success-rate` | Minimum success fraction for the run to pass (default `1.0`).           |

Each pass prints success/failure counts, latency stats (min/median/mean/max),
wall-clock time, achieved throughput (req/s) and distinct failure reasons.
The script exits non-zero if the success rate drops below `--min-success-rate`.