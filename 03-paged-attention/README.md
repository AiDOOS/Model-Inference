# 03 - Paged Attention

## Hook

Same fixed 24 MiB of KV-cache memory, same fixed 24-request workload
(6 long + 18 short prompts, all arriving at once). How many can each
approach hold *at the same time* before running out of room?

| Approach | Max concurrent seqs fit | Waste / fragmentation |
|---|---|---|
| Naive allocation | 4 | 59.6% |
| **Manual paged attention** | **11** | **8.8%** |
| vLLM | N/A on this machine - see Gotcha | N/A |

Same budget, same requests, same model. Naive allocation can only ever
have 4 sequences in flight at once; paging fits **2.75x more** - 11 - out
of the exact same 24 MiB, because it stops reserving space for tokens
that were never going to be generated.

## Mental model

- **Naive allocation** - checking a guest into a hotel by reserving them
  an entire floor, just in case they invite 50 friends up. Almost nobody
  does. The floor sits there, empty, unusable by anyone else, for as long
  as that one guest is checked in - whether they use one room or fifty.
- **Paged attention** - giving guests rooms as they actually ask for them,
  one at a time, and handing a room straight back to the front desk the
  moment its guest checks out. Nobody pre-commits to a floor.

The name is not a coincidence. This is the same idea as OS **virtual
memory paging**: a process doesn't get a contiguous slab of physical RAM
sized for the worst case it might ever touch - it gets fixed-size pages
handed out on demand, tracked in a page table, and freed the instant it
exits. Paged attention is that exact mechanism, aimed at KV-cache memory
instead of general process memory: fixed-size **blocks** instead of
pages, a **block table** instead of a page table.

## Mechanism

- **Worst-case reservation wastes memory because most sequences don't use
  their max.** Naive allocation reserved 256 tokens of context for every
  one of the 24 requests here - enough to cover the longest possible
  sequence in the workload - but the actual mix of prompt + real output
  length averaged out to using only **40.4%** of that per sequence
  (measured: 59.6% of every reserved token-slot sat empty, held for that
  sequence's entire lifetime, unusable by anyone else).
- **That waste is exactly what limits how many concurrent sequences fit -
  which directly explains module 00's naive-serving throughput problem.**
  Module 00 found naive serving's throughput stayed flat under load
  because everything just queued behind whatever was currently running.
  This module puts a number on *why* so little can run concurrently in
  the first place: if every sequence locks up a worst-case-sized slab of
  memory the instant it's admitted, a fixed memory budget can only ever
  hold a handful of them - 4, here - no matter how much spare throughput
  the CPU (or GPU) actually has sitting idle.
- **Block-based allocation with reclaiming fixes it.** Splitting the same
  24 MiB into 64 fixed-size 16-token blocks and handing them out on
  demand - just enough blocks for the prompt that actually arrived, one
  more block only when a sequence's length actually crosses into it, all
  of them back to the free list the instant a sequence finishes - fits
  11 concurrent sequences in the identical budget naive allocation could
  only fit 4 into, at 8.8% fragmentation instead of 59.6% waste.
- **Same total capacity, sliced differently.** Naive: 4 sequences x 256
  tokens = 1024 tokens of capacity. Paged: 64 blocks x 16 tokens = 1024
  tokens of capacity. Identical budget, identical token-capacity - the
  entire difference is granularity: 4 worst-case-sized chunks vs. 64
  small, independently reclaimable ones.

## Proof

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python benchmark.py
```

(`make setup && make benchmark` does the same thing, if you have GNU
`make` - this repo's dev machine doesn't, so the commands above were run
directly, same as every prior module.)

Fixed workload: 24 requests (6 long ~168-182 prompt tokens, 18 short
~41-43 prompt tokens, interleaved as 6 blocks of 1 long + 3 short), shared
`MAX_NEW_TOKENS=60`. All three approaches admit against the exact same
`MEMORY_BUDGET_BYTES = 24 MiB`. Real run, 2026-08-21:

```
                Approach  MaxConcurFit   Waste/Frag(%)    Total(s)   Tput(tok/s)
--------------------------------------------------------------------------------
        Naive allocation             4            59.6       87.52          7.81
            Manual paged            11             8.8       77.04          8.88
                    vLLM           N/A             N/A         N/A           N/A
```

vLLM row is N/A: not importable in this environment (no Windows wheels,
no usable Linux/WSL2 distro here) - see Gotcha below, and modules 01/02's
READMEs for the full explanation. The other two rows are a real, measured
run - no placeholders. Full per-request breakdown in
[RESULTS.md](RESULTS.md) and
[results/paged_attention_20260821T091959.json](results/paged_attention_20260821T091959.json).

Reading it: manual paging fits **2.75x more concurrent sequences** (11
vs. 4) in the identical memory budget, wastes **6.8x less** memory doing
it (8.8% vs. 59.6%), and - because more sequences finish per "wave"
instead of naive allocation needing six sequential waves of 4 to clear
24 requests - also finishes the whole workload faster (77.04s vs. 87.52s)
at higher throughput (8.88 vs. 7.81 tok/s), despite paging's individual
batched decode steps costing more wall-clock per step (bigger batches on
this CPU - see Gotcha).

## Gotcha

Block-based allocation is not free, and this simulation is not the real
thing:

- **Bookkeeping overhead is real, just small.** Every sequence's block
  table is a list of physical block IDs - measured here at
  `block_table_overhead_bytes = 680` total across all 24 requests (each
  entry modeled as a 4-byte block index, the same width vLLM's real block
  tables use). Tiny next to the ~24 MiB of KV cache it's managing, but
  it's not zero, and a naive per-sequence contiguous buffer has no
  equivalent structure to maintain at all.
- **Internal fragmentation is the paged version of waste, and it's real
  too - just far smaller.** A sequence that needs 61 tokens still gets
  allocated in whole 16-token blocks (4 blocks = 64 tokens here), so 3
  tokens' worth of its last block go unused. Measured aggregate: 8.8%,
  vs. naive's 59.6% - the same phenomenon, at roughly 1/16th the scale,
  because the "wasted" unit shrank from "one worst-case sequence" to "one
  block."
- **A pure admit-until-full policy genuinely deadlocks - this isn't
  theoretical, it's what the first version of `manual_paged_attention.py`
  actually did.** Admitting greedily up to the last free block, with no
  memory held back, let new admissions repeatedly out-compete
  already-running sequences for freed blocks; this workload's aggregate
  growth demand truly exceeds the 64-block pool at points in time, so an
  already-decoding sequence one token from finishing (and freeing its
  blocks for everyone else) could stall forever waiting on a block that
  admission kept handing to someone new instead. The fix - reserve a
  fixed watermark fraction of the pool (10% here) that admission can
  never touch, so already-running sequences always have room to finish -
  is the exact mechanism vLLM ships as its `watermark` config option, for
  the exact same reason.
- **Real PagedAttention (vLLM) also needs custom attention kernels to
  read from non-contiguous blocks efficiently** - our manual simulation
  still pads and stacks each active sequence's cache into one contiguous
  tensor per decode step (same technique module 02's continuous batching
  toy used), it just decides block-by-block what's *allocated*, not how
  attention *reads* it. Tracing that real kernel-level implementation is
  what module 07 (vllm-architecture) is for.
- **CPU, not GPU - same caveat as every prior module.** These are system
  RAM measurements; the same waste-vs-reclaiming dynamic applies
  identically to GPU VRAM in a real deployment, just at a scale (GBs, not
  MiBs) where a 64-block pool would instead be tens of thousands of
  blocks.
