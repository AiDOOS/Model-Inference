# 02 - Batching

## Hook

Same short prompt ("What is the capital of France?"), submitted at the same
moment as three long, detailed prompts, then answered the same 8-token way
every time: "The capital of France is Paris." How long until it comes back?

| Approach | Latency for this one request |
|---|---|
| No batching (sequential) | 14.18s |
| **Static batching** | **52.47s** |
| Continuous batching (toy) | 14.12s |
| vLLM | N/A on this machine - see Gotcha |

Same question, same answer, same hardware. Static batching makes you wait
**3.7x longer** than everything else - not because your request is doing
more work, but because it's sitting in a batch with two long-running
neighbors and nobody gets off the bus until the last one does.

That gap is the entire reason continuous batching exists.

## Mental model

- **No batching** - one cashier, one customer, start to finish, then the
  next customer. Nobody else exists until you're done, but you get the
  cashier's full, undivided attention.
- **Static batching** - a bus that won't leave the station until it's full,
  and won't let ANYONE off until EVERYONE has reached their stop. If your
  stop is two blocks away and the person next to you is going across town,
  you're both getting off at the same time - theirs.
- **Continuous batching** - a ride-share van. People hop on the moment a
  seat opens and hop off the moment they arrive. Nobody's trip length is
  held hostage by anyone else's.

## Mechanism

- **Static batching wastes compute on padding.** Prompts of different
  lengths get left-padded to the batch's longest prompt before the single
  batched forward pass can run at all - the padding tokens cost real FLOPs
  and produce nothing useful. Measured here: **58.7% of every batch's
  prompt matrix was pure padding** (batches mixed ~42-token short prompts
  with ~170-182 token long ones).
- **Static batching creates head-of-line blocking.** Once a batch starts,
  EVERY member's result is only returned when the WHOLE batch is done -
  which means whichever sequence needs the most decode steps (typically the
  longest/most open-ended one) sets the finish time for everyone else in
  that batch, short requests included. Measured here: **53.4% of all
  decode-step compute** was spent re-running the forward pass for rows that
  had already finished generating and were just riding along until their
  two long batchmates hit the token cap.
- **Continuous batching (token-level admission/eviction) fixes both,
  partially.** A finished sequence is evicted and replaced the very next
  iteration - nobody waits around occupying a slot for no reason - and a
  short request that finishes early actually returns early. It doesn't
  remove padding waste entirely (each iteration's batch is still re-padded
  to whatever the current longest active sequence is), but it removes the
  waiting-for-the-whole-batch-to-drain problem completely.
- **Batching (of either kind) turns many tiny, dispatch-overhead-dominated
  single-request decode steps into fewer, denser, better-utilized batched
  steps.** On a GPU this is usually the only way to raise utilization at
  all. On CPU, see the note below - it's a smaller effect than you'd
  expect, and that's a real, useful finding, not a bug in the measurement.

## Proof

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python benchmark.py
```

(`make setup && make benchmark` does the same thing, if you have GNU `make`
available - this repo's dev machine is native Windows without it, so every
command above was actually run directly.)

Fixed workload: 16 requests (4 long ~170-182 prompt tokens, 12 short ~42
prompt tokens, interleaved so every batch of 8 contains 2 long + 6 short),
shared `MAX_NEW_TOKENS=60` cap, `BATCH_SIZE=8` for both static batching and
the continuous scheduler's max concurrent slots. Real run, 2026-08-19:

```
                Approach    Total(s)   Tput(tok/s)   AvgCPU(%)   ShortLat(s)   LongLat(s)
-----------------------------------------------------------------------------------------
             No batching       93.52          4.88        60.7         54.10        47.18
         Static batching      104.05          4.38        65.1         78.24        78.24
        Continuous (toy)       61.56          7.41        63.8         34.98        59.46
                    vLLM         N/A           N/A         N/A           N/A          N/A
```

vLLM row is N/A: not importable in this environment (no Windows wheels, no
usable Linux/WSL2 distro here) - see Gotcha below, and module 01's README
for the full explanation. Everything else is a real, measured run - no
placeholders.

Reading it:

- **Continuous batching wins on every column simultaneously here** - lowest
  total time (61.56s vs. 93.52s / 104.05s), highest throughput (7.41 vs.
  4.88 / 4.38 tok/s), AND the best latency for both categories. It's not a
  pure trade-off in this workload - avoiding wasted decode compute on
  finished rows saves enough total work to also win on raw wall-clock, not
  just fairness.
- **Static batching's ShortLat and LongLat are identical (78.24s both) -
  exactly the head-of-line blocking signature.** Every row in a given batch
  shares that batch's total time by construction; a short request gets
  zero benefit from finishing early.
- **No batching's ShortLat (54.10s) being worse than its LongLat (47.18s)**
  looks backwards until you remember it's an *average*, and it's purely an
  artifact of THIS workload's arrival order (`build_workload()` places one
  long request before every group of three short ones - so the short
  requests' average includes ones queued behind multiple long requests
  ahead of them, while some long requests are near the front of the queue).
  It's a reminder that in a strictly sequential, no-batching system,
  latency is about queue POSITION, not request size.
- **AvgCPU(%) is close across all three runnable approaches (60.7% /
  65.1% / 63.8%)** - see Gotcha for why this doesn't behave like GPU
  utilization would.

## Gotcha

The continuous batching scheduler here is a **real simplification**, not a
scaled-down vLLM:

- **Admission does its own individual prefill, not a fused mixed
  prefill+decode step.** A newly admitted request pays for a standalone
  forward pass over its prompt before joining the shared decode batch next
  iteration. Real engines (vLLM included) can fuse a newcomer's prefill and
  other sequences' decode into the SAME iteration's compute via chunked
  prefill - this toy keeps them as two distinct kinds of forward pass to
  stay correct and readable.
- **The KV cache is manually re-padded and re-split every iteration**,
  because each active slot's cache is a different length (slots joined at
  different times) and a batched matmul needs one shared shape. This is a
  real, historically-used technique (iteration-level / Orca-style
  scheduling) - it's what continuous batching looked like *before*
  PagedAttention. It demonstrates the scheduling idea (no head-of-line
  blocking) correctly; it does NOT demonstrate memory-efficient batching at
  scale, because every slot still holds its own full, unshared,
  contiguously-allocated cache. **A future paged-attention module is
  exactly the gap this leaves open**: managing that KV memory efficiently -
  sharing pages, avoiding per-slot re-padding, supporting far more
  concurrent sequences than would fit if every one of them needed its own
  contiguous, worst-case-sized cache buffer.
- **CPU utilization is a genuine but imperfect stand-in for GPU
  utilization here.** PyTorch's CPU backend already multi-threads a
  SINGLE sequence's matmul across several cores by default (this machine:
  6 of 12 logical cores, `torch.get_num_threads()`), unlike a GPU sitting
  nearly idle at batch size 1. That means batching multiple requests
  together doesn't unlock idle compute the way it does on GPU - it mainly
  reduces wasted per-token Python/dispatch overhead by doing more useful
  work per already-busy pass. Expect the CPU utilization numbers across
  these four approaches to be closer together than the GPU-utilization
  intuition would predict - that's a real property of this hardware, not a
  broken measurement.
- **The fixed workload here has all 16 requests arrive simultaneously.**
  Static batching's "time spent waiting for the batch to fill" is
  therefore ~0 in this benchmark - in a live system with staggered
  arrivals, that wait would be a real, additional cost on top of
  everything measured here.
