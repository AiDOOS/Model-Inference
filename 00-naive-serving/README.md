# 00 - Naive Serving

Baseline module. Runs Qwen2.5-0.5B-Instruct on CPU behind the simplest
possible HTTP server, then benchmarks it under concurrency to put two
numbers on the table:

1. **Naive serving collapses under concurrency.** One model, no batching,
   no scheduler: every request queues behind the last one. Latency grows
   roughly linearly with concurrency while throughput stays flat instead of
   scaling up.
2. **Prefill and decode are different phases with different cost
   profiles.** Prefill is one parallel forward pass over the whole prompt.
   Decode is N sequential single-token forward passes. Measuring them
   separately (instead of calling `model.generate()` and getting one
   opaque latency number back) is the whole point of this module.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`torch` is installed separately with the CPU wheel index so `pip` doesn't try
to pull a CUDA build you don't need on a CPU-only box like this one's.

> **Windows: `WinError 206` / "filename or extension is too long" while
> installing torch?** Torch ships license files nested many folders deep, and
> Windows caps full paths at 260 characters by default. If your `.venv` lives
> somewhere with a long path (deeply nested folders, spaces in folder names,
> etc.), the combined path can blow past that limit. Two fixes, pick one:
> - **Enable long paths once, system-wide** (recommended - fixes this for
>   every project, forever). In an **admin** PowerShell:
>   ```powershell
>   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
>   ```
>   No reboot needed. Then re-run the `pip install` commands above.
> - **Or put the venv somewhere short**, e.g. `python -m venv C:\mi-venv-00`,
>   then activate and install from there instead of `.venv`.

## Run

Terminal 1 - start the server (loads the model once, stays resident):

```bash
python server.py
```

The first run downloads Qwen2.5-0.5B-Instruct from the Hugging Face Hub
(a few hundred MB) - expect it to take a minute or two before you see
`Application startup complete`. Later runs load from the local cache and
start in a few seconds.

Terminal 2 - run the benchmark once the server logs that the model is loaded:

```bash
python benchmark.py
```

The default run (concurrency 1/4/8/16/32, 64 tokens each) can take several
minutes on a CPU - the server processes every request strictly one at a time
by design (see below), so the concurrency=32 batch alone is running 32
requests back-to-back on a single stream. For a quick sanity check first,
shrink it:

```bash
python benchmark.py --levels 1 2 4 --max-new-tokens 16
```

## What server.py actually measures

`POST /generate` does not call `model.generate()`. It does the two phases by
hand:

- **Prefill**: `model(input_ids, use_cache=True)` once, over the full
  prompt. Timed as a single number, `prefill_time_s` - this is effectively
  time-to-first-token.
- **Decode**: a loop, `model(one_token, past_key_values=..., use_cache=True)`,
  timed on every single step. Returned as `decode_times_s`, a list - not an
  average - so you can see per-token latency directly instead of a number
  that's already had the interesting part averaged away.

Note `len(decode_times_s) == output_tokens - 1`: the first output token is a
free byproduct of the prefill pass (its logits already predict token 1), so
the decode loop only runs for every token after that.

A single `threading.Lock` wraps prefill *and* the entire decode loop for one
request. Only one request is ever inside that lock. Everything else - even
though FastAPI happily accepts many concurrent HTTP connections - sits
waiting. `queue_wait_s` is exactly that wait, measured before the lock is
acquired. This is the naive part, and it's deliberate: nothing here batches
concurrent requests together or interleaves their decode steps. That's what
a real inference server (and a later module in this repo) has to fix.

## What benchmark.py measures

For concurrency levels 1, 4, 8, 16, 32, it fires that many concurrent
requests, waits for all of them, and reports:

| Column | Meaning |
|---|---|
| `AvgPrefill(s)` | mean `prefill_time_s` across the batch - server compute only |
| `AvgDecode/tok(s)` | mean of every decode step across the batch - server compute only |
| `AvgQueueWait(s)` | mean time each request spent waiting for the lock |
| `AvgE2E(s)` | mean total client-observed latency per request |
| `Tput(tok/s)` | total output tokens in the batch / wall-clock time for the whole batch |

Expected shape of the results:

- `AvgPrefill` and `AvgDecode/tok` should stay roughly **flat** across
  concurrency levels - whichever request holds the lock always has the CPU
  to itself, so its own compute cost doesn't change.
- `AvgQueueWait` and `AvgE2E` should **grow with concurrency** - that's
  requests piling up behind each other.
- `Tput(tok/s)` should stay roughly **flat**, not scale up with
  concurrency - proof that adding concurrent load doesn't buy you more
  capacity here, only worse per-request latency.

If prefill and decode-per-token look identical at concurrency=1, that's a
smell, not a win - go back and check prompt length vs. `max_new_tokens`.
Prefill cost scales with prompt length (one pass over all prompt tokens at
once); decode-per-token cost is per-single-token and should be noticeably
cheaper per step, though it accumulates over many sequential steps.
