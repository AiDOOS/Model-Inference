# Results

## Environment

- CPU-only, Windows 10, same machine as modules 00/01/02.
- Qwen/Qwen2.5-0.5B-Instruct, float32, cached locally.
- Measured (not formula-derived) KV-cache size: `bytes_per_token = 24576.0`
  (24 KB/token, K+V across all 24 layers) - obtained by running one real
  forward pass and inspecting the resulting cache tensors' shapes and
  dtype (see `kv_cache.py::measure_kv_bytes_per_token`).
- `MEMORY_BUDGET_BYTES = 24 MiB`, identical for every approach.
  `MAX_SEQ_LEN_NAIVE = 256` tokens -> naive's per-sequence reservation is
  exactly 6.00 MB, so `24 MiB / 6 MB = 4` concurrent sequences, by
  construction. `BLOCK_SIZE_TOKENS = 16` -> `24 MiB / (16 x 24576 bytes)
  = 64` total blocks for paging.
- vLLM: **not runnable**, same reason established in modules 01/02 - zero
  Windows wheels on PyPI, no usable Linux/WSL2 distro on this machine.
  `vllm_paged.py` detects this and skips cleanly.

## Full benchmark (2026-08-21)

```
python benchmark.py
```

Fixed workload: 24 requests, 6 long (~168-182 prompt tokens) + 18 short
(~41-43 prompt tokens), arranged as 6 blocks of (1 long, 3 short), all
arriving simultaneously. Shared `MAX_NEW_TOKENS=60`.

```
                Approach  MaxConcurFit   Waste/Frag(%)    Total(s)   Tput(tok/s)
--------------------------------------------------------------------------------
        Naive allocation             4            59.6       87.52          7.81
            Manual paged            11             8.8       77.04          8.88
                    vLLM           N/A             N/A         N/A           N/A
```

Full raw output saved to
[results/paged_attention_20260821T091959.json](results/paged_attention_20260821T091959.json).

### Naive allocation's waste, measured directly

```
bytes_per_token           = 24576.0   (24.0 KB/token, measured)
max_seq_len_reservation   = 256 tokens  ->  6.00 MB reserved per sequence, always
peak_concurrent           = 4
waste_percent             = 59.6%   (share of every reserved token-slot never actually used)
```

Every one of the 24 requests reserved the full 256-token slab on
admission, regardless of whether it went on to use 50 tokens or 242.
`total_reserved_tokens = 24 x 256 = 6144`; `total_used_tokens = 2480`
(sum of real prompt+output length across all 24) -> `1 - 2480/6144 =
59.6%` wasted, continuously, for as long as each sequence was held.

### Manual paged attention, measured directly

```
total_blocks                  = 64   (24 MiB / (16 tok x 24576 bytes/tok))
peak_concurrent                = 11
peak_blocks_in_use             = 64  (pool was fully committed at peak)
stalls                          = 0
internal_fragmentation_percent = 8.8%
block_table_overhead_bytes     = 680   (170 total blocks ever held x 4 bytes/entry)
```

`total_capacity_tokens = 2720` (sum of `blocks_held_at_finish x 16`
across all 24 requests); same `total_used_tokens = 2480` as naive ->
`1 - 2480/2720 = 8.8%` fragmentation - the same phenomenon as naive's
waste, at roughly 1/7th the scale, because the wasted unit shrank from
"one worst-case sequence" (256 tokens) down to "the unused tail of one
16-token block."

**Getting to `stalls = 0` took a real fix, not just tuning.** The first
version of this scheduler admitted new requests greedily until the pool
was full, with no memory held back for already-running sequences to grow
into - and it deadlocked in practice (hit `MAX_ITERATIONS` with the queue
still non-empty), because this workload's peak aggregate block demand
genuinely exceeds 64 at points in time, and admission kept winning the
race for every block eviction freed. The fix - `PAGED_WATERMARK_FRACTION
= 0.10`, a slice of the pool admission may never touch - is modeled on
vLLM's real `watermark` config option, which exists for the identical
reason. See README Gotcha for the full story.

### Per-request detail, both approaches

```
                                naive_allocation              manual_paged_attention
 id  cat  prompt  out  actual   reserved  waste  latency_s     blocks  capacity  frag  latency_s
  0  long   182   60    242       256      14     33.26          16      256     14     45.84
  1  short   42    8     50       256      206     6.64           4       64     14     15.38
  2  short   41   28     69       256      187    17.21           5       80     11     27.49
  3  short   43   18     61       256      195    11.95           4       64      3     22.00
  4  long   168   60    228       256      28     36.60          15      240     12     45.84
  5  short   42    8     50       256      206    15.36           4       64     14     15.38
  6  short   41   28     69       256      187    28.62           5       80     11     27.49
  7  short   43   18     61       256      195    24.39           4       64      3     22.00
  8  long   170   60    230       256      26     56.12          15      240     10     45.84
  9  short   42    8     50       256      206    32.08           4       64     14     15.38
 10  short   41   28     69       256      187    47.02           5       80     11     27.49
 11  short   43   18     61       256      195    42.06           4       64      3     26.50
 12  long   182   60    242       256      14     66.68          16      256     14     71.41
 13  short   42    8     50       256      206    45.47           4       64     14     34.53
 14  short   41   28     69       256      187    59.54           5       80     11     44.19
 15  short   43   18     61       256      195    54.20           4       64      3     66.55
 16  long   168   60    228       256      28     83.47          15      240     12     77.04
 17  short   42    8     50       256      206    59.54           4       64     14     59.11
 18  short   41   28     69       256      187    74.35           5       80     11     71.41
 19  short   43   18     61       256      195    69.04           4       64      3     66.55
 20  long   170   60    230       256      26     87.52          15      240     10     77.04
 21  short   42    8     50       256      206    72.42           4       64     14     59.11
 22  short   41   28     69       256      187    83.47           5       80     11     71.41
 23  short   43   18     61       256      195    81.66           4       64      3     66.55
```

**Reading it:**

- **Naive processes 24 requests in exactly 6 sequential waves of 4** -
  request 20's 87.52s latency is essentially the whole run's wall-clock,
  because it's queued behind 5 waves of admission-limited concurrency
  ahead of it. Paged admits its first 11 requests in one wave (ids 0-10,
  latencies 15-46s) and the remaining 13 in a second (ids 11-23), so the
  tail request (16 or 20, at 77.04s) finishes **10.5s sooner** overall.
- **Early short requests actually finish *slower* under paging than
  under naive (e.g. id 1: 6.64s naive vs. 15.38s paged), the opposite of
  what module 02's continuous batching found.** Paging's first wave packs
  11 sequences into one batched decode step; naive's wave only ever holds
  4. Bigger batches cost more wall-clock per step on this CPU (same
  caveat module 02's Gotcha measured: PyTorch's CPU backend already
  multi-threads a single sequence's matmul across multiple cores, so
  batching more sequences together doesn't unlock idle compute the way
  it would on GPU - it mainly adds more total work to one step). Paging
  still wins on *total* wall-clock and throughput because it needs fewer
  waves overall to clear the same 24 requests - it trades slightly worse
  early-latency for a shorter whole-workload makespan, a genuinely
  different trade-off than continuous batching's "wins on every column"
  result in module 02.
- **`waste`/`frag` columns tell the real story independent of timing:**
  every naive row wastes 187-206 of its 256 reserved tokens; every paged
  row fragments only 3-14 tokens out of its much smaller, actually-needed
  allocation.

## vLLM (2026-08-21)

```
python vllm_paged.py
```

```
[vllm_paged] SKIPPED - vLLM cannot run in this environment.
[vllm_paged] reason: vLLM is not importable in this environment (No module
named 'vllm'). Running on Windows 10. Same limitation established in
modules 01 and 02: vLLM ships no Windows wheels and this machine has no
usable Linux/WSL2 distro.
[vllm_paged] no numbers are reported below - honest result, not a bug.
```

No fabricated vLLM numbers exist anywhere in this module. If this repo is
ever run from Linux or a real WSL2 distro, `pip install vllm` followed by
`python vllm_paged.py` / `python benchmark.py` will produce real
`vllm_paged` numbers - including vLLM's own real block-manager accounting
(`cache_config.block_size` / `num_gpu_blocks`, configured via
`VLLM_CPU_KVCACHE_SPACE` to the identical 24 MiB budget) - without any
code changes.
