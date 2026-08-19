# Results

## Environment

- CPU-only, 12GB RAM, Windows 10, same machine as modules 00/01.
- Qwen/Qwen2.5-0.5B-Instruct, float32, cached locally.
- `torch.get_num_threads() == 6` (of 12 logical cores) - PyTorch's default
  intra-op thread pool already spreads a SINGLE sequence's matmul across
  multiple cores. This matters for reading the CPU utilization numbers
  below - see README "Gotcha".
- vLLM: **not runnable**, same reason established in module 01 - zero
  Windows wheels on PyPI, no usable Linux/WSL2 distro on this machine.
  `vllm_batching.py` detects this and skips cleanly.

## Full benchmark (2026-08-19)

```
python benchmark.py
```

Fixed workload: 16 requests, 4 long (~170-182 prompt tokens) + 12 short
(~42 prompt tokens), arranged as 4 blocks of (1 long, 3 short) so every
static batch of 8 contains 2 long + 6 short. Shared `MAX_NEW_TOKENS=60`.

```
                Approach    Total(s)   Tput(tok/s)   AvgCPU(%)   ShortLat(s)   LongLat(s)
-----------------------------------------------------------------------------------------
             No batching       93.52          4.88        60.7         54.10        47.18
         Static batching      104.05          4.38        65.1         78.24        78.24
        Continuous (toy)       61.56          7.41        63.8         34.98        59.46
                    vLLM         N/A           N/A         N/A           N/A          N/A
```

Full raw output (including the CPU utilization time-series) saved to
[results/batching_20260819T195513.json](results/batching_20260819T195513.json).

### Static batching's waste, measured directly

```
avg_padding_ratio      = 58.6%   (fraction of the padded prompt matrix that was left-padding, not real tokens)
avg_decode_waste_ratio = 53.4%   (fraction of row x decode-step slots spent on rows already finished)
```

Both batches (8 requests each) padded every prompt up to 182 tokens (the
longest prompt's real length) and ran the full 59-step decode loop because
2 of the 8 rows per batch never hit EOS before the shared 60-token cap.

### Per-request latency, all three runnable approaches

```
                 no_batching        static_batching      continuous_batching_toy
 id  cat   out    latency_s           latency_s               latency_s
  0  long   60      12.20                52.47                   58.31
  1  short   8      14.18                52.47                   14.12
  2  short  28      19.59                52.47                   36.12
  3  short  18      23.40                52.47                   25.36
  4  long   60      35.47                52.47                   58.31
  5  short   8      37.45                52.47                   14.12
  6  short  28      43.01                52.47                   36.12
  7  short  18      46.77                52.47                   25.36
  8  long   60      58.77               104.02                   59.68
  9  short   8      60.78               104.02                   22.12
 10  short  28      66.26               104.02                   48.83
 11  short  18      70.14               104.02                   43.47
 12  long   60      82.29               104.02                   61.56
 13  short   8      84.30               104.02                   43.47
 14  short  28      89.78               104.02                   56.14
 15  short  18      93.52               104.02                   54.50
```

**Reading it:**

- **Static batching's latency column has exactly two values: 52.47s and
  104.02s.** Every request in batch 1 (ids 0-7) waits exactly 52.47s,
  whether it's an 8-token answer or a 60-token one - because static
  batching literally cannot return any result before the whole batch's
  loop exits. Same for batch 2 (ids 8-15) at 104.02s. This is the cleanest
  possible demonstration of head-of-line blocking: request length has
  **zero** effect on this column.
- **Requests 1 and 5 - both "What is the capital of France?", both an
  8-token answer - finish at 14.12s under continuous batching, essentially
  identical to no_batching's 14.18s/37.45s-if-it-were-first.** They get
  admitted into the very first round of slots (alongside the two long
  requests from that block) and are evicted the moment they're done,
  without waiting for their long batchmates. Under static batching, the
  same requests take 52.47s - a **3.7x** latency penalty purely from the
  scheduling policy, not from any difference in the work itself.
- **Continuous batching also wins on total wall-clock (61.56s vs. 93.52s /
  104.05s) and throughput (7.41 vs. 4.88 / 4.38 tok/s)** - in this
  workload, skipping 53.4% wasted decode compute saves enough total work
  to win outright, not just trade fairness for throughput.
- **No batching's numbers climb in strict queue order** (12.20 -> 93.52,
  request 15 waits for all 15 requests ahead of it) - the baseline's
  problem was never "slow individual requests," it's "everything queues."

## vLLM (2026-08-19)

```
python vllm_batching.py
```

```
[vllm_batching] SKIPPED - vLLM cannot run in this environment.
[vllm_batching] reason: vLLM is not importable in this environment (No module
named 'vllm'). Running on Windows 10. Same limitation established in module 01:
vLLM ships no Windows wheels and this machine has no usable Linux/WSL2 distro.
[vllm_batching] no numbers are reported below - honest result, not a bug.
```

No fabricated vLLM numbers exist anywhere in this module. If this repo is
ever run from Linux or a real WSL2 distro, `pip install vllm` followed by
`python vllm_batching.py` / `python benchmark.py` will produce real
`vllm_total_time_s` and per-request latency numbers without any code
changes.
