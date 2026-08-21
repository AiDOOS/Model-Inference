"""
Module 03 - Approach 2: Manual paged KV-cache allocation (the core idea).

This is the "give guests rooms as they actually need them, and reclaim a
room the moment a guest checks out" approach from the README - and it's a
deliberately close simulation of the actual data structure vLLM's real
PagedAttention uses: memory is divided into fixed-size BLOCKS (here,
BLOCK_SIZE_TOKENS=16 tokens each), a free list hands out block IDs on
demand, and every sequence owns a BLOCK TABLE - just a list of the
physical block IDs currently assigned to it, in order.

The name is not a coincidence: this is the same idea as OS virtual memory
paging. A process doesn't get a single contiguous slab of physical RAM
sized for its worst-case memory use; it gets pages handed out as it
touches new memory, mapped through a page table, and freed back to the
system the moment it exits. Swap "process" for "sequence," "page" for
"block," and "page table" for "block table" and you have this file.

Concretely, per sequence:
  - ADMISSION allocates only ceil(prompt_tokens / BLOCK_SIZE_TOKENS)
    blocks - enough for the PROMPT that actually arrived, not for
    whatever the generation might eventually grow into.
  - GROWTH: every time decoding crosses a block boundary (cur_len becomes
    an exact multiple of BLOCK_SIZE_TOKENS), one more block is pulled from
    the free list and appended to that sequence's block table. If none is
    free, the sequence STALLS for this iteration - it sits out of the
    batch until some other sequence frees a block, instead of being
    allowed to overrun into memory it doesn't own.
  - EVICTION frees every block in a finished sequence's block table back
    to the free list immediately - available to the very next admission
    or growth request, not held onto "just in case."

Same admit/evict/batched-decode-step loop shape as module 02's
continuous_batching_toy.py and this module's own naive_allocation.py - the
memory-accounting policy below is the only thing that's different.
"""

import argparse
import math
import time
from typing import Dict, List, Optional

import torch

from kv_cache import (
    BLOCK_SIZE_TOKENS,
    MAX_ITERATIONS,
    MAX_NEW_TOKENS,
    MEMORY_BUDGET_BYTES,
    PAGED_WATERMARK_FRACTION,
    build_input_ids,
    build_workload,
    eos_ids_for,
    load_model,
    measure_kv_bytes_per_token,
    run_batched_decode_step,
)

BLOCK_TABLE_ENTRY_BYTES = 4  # one block ID stored as a uint32, same width vLLM's real block tables use


class PagedMemoryPool:
    """A free list of block IDs (0..total_blocks-1) plus per-sequence block
    tables. Blocks are fungible - any free block ID is as good as any
    other - so the free list is just a Python list used as a stack: pop to
    allocate, append to free. This is the actual bookkeeping structure
    real paged KV-cache managers use, just without a real physical tensor
    pool behind it (see README Gotcha for what that leaves out)."""

    def __init__(self, budget_bytes: int, block_size_tokens: int, bytes_per_token: float,
                 watermark_fraction: float = PAGED_WATERMARK_FRACTION):
        self.block_size_tokens = block_size_tokens
        self.block_bytes = block_size_tokens * bytes_per_token
        self.total_blocks = int(budget_bytes // self.block_bytes)
        self.free_block_ids: List[int] = list(range(self.total_blocks))
        # Blocks admission may never touch, reserved for sequences that are
        # ALREADY running to grow into. Without this, admission is happy to
        # take the pool right up to its last free block for brand-new
        # requests - and since this workload's aggregate growth demand
        # genuinely exceeds this pool's total size at points in time (long
        # requests alone eventually need more blocks than start out free),
        # admitting that aggressively starves already-running sequences of
        # the ONE more block they need to ever finish and free anything -
        # a real deadlock, verified by hitting MAX_ITERATIONS with this
        # reserve set to 0. vLLM has the identical knob, called
        # `watermark`, for the identical reason.
        self.reserve_blocks = round(watermark_fraction * self.total_blocks)
        self.peak_blocks_in_use = 0
        self.peak_concurrent = 0
        self.stalls = 0  # count of (sequence, iteration) growth requests that had to wait

    def blocks_needed_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size_tokens)

    def can_admit(self, n: int) -> bool:
        """Admission-only check: would taking n blocks dip into the
        reserve? Growth never calls this - growth is exactly what the
        reserve exists to protect, so it's always allowed to use every
        block still physically free, including the reserved ones."""
        return len(self.free_block_ids) - n >= self.reserve_blocks

    def allocate(self, n: int) -> Optional[List[int]]:
        """Try to pull n block IDs off the free list. Returns None (and
        allocates NOTHING - no partial allocation) if fewer than n are
        free; caller decides whether that means "stay queued" (admission)
        or "stall this iteration" (growth)."""
        if len(self.free_block_ids) < n:
            return None
        allocated = self.free_block_ids[-n:]
        del self.free_block_ids[-n:]
        blocks_in_use = self.total_blocks - len(self.free_block_ids)
        self.peak_blocks_in_use = max(self.peak_blocks_in_use, blocks_in_use)
        return allocated

    def free(self, block_ids: List[int]) -> None:
        """Return blocks to the free list immediately - the instant this
        sequence is done with them, they're available to admit the very
        next queued request or grow some other active sequence, in the
        SAME iteration if needed."""
        self.free_block_ids.extend(block_ids)

    def note_concurrency(self, active_count: int) -> None:
        self.peak_concurrent = max(self.peak_concurrent, active_count)


def run_manual_paged_attention(
    max_new_tokens: int = MAX_NEW_TOKENS,
    block_size_tokens: int = BLOCK_SIZE_TOKENS,
    budget_bytes: int = MEMORY_BUDGET_BYTES,
) -> Dict:
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)
    num_layers = model.config.num_hidden_layers
    bytes_per_token = measure_kv_bytes_per_token(tokenizer, model)

    pool = PagedMemoryPool(budget_bytes, block_size_tokens, bytes_per_token)
    queue = list(build_workload())
    active: List[Dict] = []
    results = []
    admitted_over_time = []

    wall_start = time.perf_counter()

    for iteration in range(MAX_ITERATIONS):
        # --- EVICT: free this sequence's ENTIRE block table the instant
        # it finishes - every block it held becomes available to anyone
        # else this same iteration. ---
        still_active = []
        for slot in active:
            if slot["finished"]:
                pool.free(slot["block_table"])
                text = tokenizer.decode(slot["generated"], skip_special_tokens=True)
                actual_len = slot["prompt_tokens"] + len(slot["generated"])
                capacity_tokens = len(slot["block_table"]) * block_size_tokens
                results.append({
                    "request_id": slot["request_id"],
                    "category": slot["category"],
                    "prompt_tokens": slot["prompt_tokens"],
                    "output_tokens": len(slot["generated"]),
                    "actual_len_tokens": actual_len,
                    "blocks_held_at_finish": len(slot["block_table"]),
                    "capacity_tokens": capacity_tokens,
                    "internal_fragmentation_tokens": capacity_tokens - actual_len,
                    "latency_s": time.perf_counter() - wall_start,
                    "text": text,
                })
            else:
                still_active.append(slot)
        active = still_active

        if not active and not queue:
            break

        # --- GROWTH: any active sequence whose next token would cross a
        # block boundary (its current length is an exact multiple of
        # BLOCK_SIZE_TOKENS) needs ONE more block before it can take that
        # step. If the pool has one, grow; if not, this sequence sits out
        # of THIS iteration's batch - it stalls, exactly like a real paged
        # allocator declining to let a process write past a page it
        # doesn't own, rather than corrupting the next block over.
        #
        # THIS RUNS BEFORE ADMISSION, DELIBERATELY. An earlier version of
        # this scheduler admitted new requests first and let growth take
        # whatever blocks were left over - and deadlocked: with the queue
        # rarely empty, every block freed by an eviction got grabbed by a
        # brand-new prompt before an already-decoding sequence one token
        # from finishing ever got a turn, so that sequence (and whoever
        # was queued behind waiting on IT to free its blocks) stalled
        # forever. Already-RUNNING sequences must have first claim on
        # freed memory over WAITING ones - the same preemption priority
        # real schedulers (vLLM included) use for exactly this reason. ---
        runnable = []
        for slot in active:
            if slot["cur_len"] % block_size_tokens == 0:
                new_block = pool.allocate(1)
                if new_block is None:
                    pool.stalls += 1
                    continue  # sits out this iteration; retried next iteration
                slot["block_table"].append(new_block[0])
            runnable.append(slot)

        if runnable:
            new_next_tokens, trimmed_caches, real_lens = run_batched_decode_step(model, runnable, num_layers)
            for j, slot in enumerate(runnable):
                slot["cache"] = trimmed_caches[j]
                slot["cur_len"] = real_lens[j]
                token_id = new_next_tokens[j].item()
                slot["generated"].append(token_id)
                slot["next_token"] = new_next_tokens[j].view(1, 1)
                if token_id in eos_ids or len(slot["generated"]) >= max_new_tokens:
                    slot["finished"] = True

        # --- ADMIT: only AFTER running sequences' growth needs this
        # iteration are settled do leftover free blocks - ABOVE the
        # watermark reserve - get offered to the queue. Enough for that
        # request's PROMPT length, not any worst case: a 41-token short
        # prompt asks for ceil(41/16)=3 blocks; a 182-token long prompt
        # asks for ceil(182/16)=12. Naive allocation would have reserved
        # 256 for both. ---
        while queue:
            req = queue[0]
            input_ids = build_input_ids(tokenizer, req["prompt"])
            prompt_tokens = input_ids.shape[1]
            blocks_needed = pool.blocks_needed_for(prompt_tokens)
            if not pool.can_admit(blocks_needed):
                break  # admitting this would dip into the watermark reserve - stays queued
            block_table = pool.allocate(blocks_needed)
            if block_table is None:
                break  # not enough free blocks for even this prompt right now - stays queued
            queue.pop(0)
            with torch.inference_mode():
                out = model(input_ids, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)
            generated = [next_token.item()]
            active.append({
                "request_id": req["request_id"],
                "category": req["category"],
                "prompt_tokens": prompt_tokens,
                "generated": generated,
                "cache": out.past_key_values,
                "cur_len": prompt_tokens,
                "next_token": next_token,
                "block_table": block_table,
                "finished": generated[-1] in eos_ids or len(generated) >= max_new_tokens,
            })

        pool.note_concurrency(len(active))
        admitted_over_time.append({
            "iteration": iteration, "active_count": len(active),
            "blocks_in_use": pool.total_blocks - len(pool.free_block_ids),
        })
    else:
        raise RuntimeError(
            f"manual_paged_attention hit MAX_ITERATIONS={MAX_ITERATIONS} without draining "
            "the queue - likely a scheduler bug, not real workload size."
        )

    total_time_s = time.perf_counter() - wall_start
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_capacity_tokens = sum(r["capacity_tokens"] for r in results)
    total_used_tokens = sum(r["actual_len_tokens"] for r in results)
    total_blocks_ever_held = sum(r["blocks_held_at_finish"] for r in results)

    return {
        "approach": "manual_paged_attention",
        "budget_bytes": budget_bytes,
        "bytes_per_token": bytes_per_token,
        "block_size_tokens": block_size_tokens,
        "total_blocks": pool.total_blocks,
        "peak_concurrent": pool.peak_concurrent,
        "peak_blocks_in_use": pool.peak_blocks_in_use,
        "stalls": pool.stalls,
        "internal_fragmentation_percent": 100.0 * (1 - total_used_tokens / total_capacity_tokens),
        "block_table_overhead_bytes": total_blocks_ever_held * BLOCK_TABLE_ENTRY_BYTES,
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "admitted_over_time": admitted_over_time,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("[manual_paged_attention] loading model + workload ...", flush=True)
    summary = run_manual_paged_attention(args.max_new_tokens)

    print(f"[manual_paged_attention] total_blocks = {summary['total_blocks']}")
    print(f"[manual_paged_attention] peak_concurrent = {summary['peak_concurrent']}")
    print(f"[manual_paged_attention] peak_blocks_in_use = {summary['peak_blocks_in_use']}")
    print(f"[manual_paged_attention] stalls = {summary['stalls']}")
    print(f"[manual_paged_attention] internal_fragmentation_percent = "
          f"{summary['internal_fragmentation_percent']:.1f}%")
    print(f"[manual_paged_attention] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[manual_paged_attention] throughput_tok_s = {summary['throughput_tok_s']:.2f}")

    for r in sorted(summary["results"], key=lambda r: r["request_id"]):
        print(f"  [{r['request_id']:2d}] {r['category']:5s} "
              f"actual={r['actual_len_tokens']:3d} blocks={r['blocks_held_at_finish']:2d} "
              f"capacity={r['capacity_tokens']:3d} frag={r['internal_fragmentation_tokens']:2d} "
              f"latency={r['latency_s']:.3f}s")


if __name__ == "__main__":
    main()
