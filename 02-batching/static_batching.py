"""
Module 02 - Approach 2: Static batching.

Collect BATCH_SIZE requests, left-pad all their prompts to the same width,
run ONE batched prefill + decode loop over the whole padded batch, and only
return ANY result once the WHOLE batch is done. This is the "bus that
won't leave until full, and won't let anyone off until everyone reaches
their stop" model from the README.

Implemented as a hand-rolled batched forward-pass loop (mirroring the style
of every other module in this repo) rather than a single `model.generate()`
call, because the interesting numbers here - per-row "would have finished
here" step, padding waste, wasted decode compute on already-finished rows -
need per-step visibility that a plain `.generate()` call doesn't expose
without extra plumbing anyway. The BEHAVIOR is identical to what
`.generate()` does internally for a padded batch: every row marked
"finished" (hit EOS) keeps riding along through the batched matmul every
subsequent step - fed a dummy pad token - because the tensor shapes are
shared across the whole batch. Nobody gets off early.

Two distinct kinds of waste are measured:
  - padding_ratio: fraction of the PROMPT matrix that's just left-padding
    (short prompts padded up to the batch's longest prompt).
  - decode_waste_ratio: fraction of (row x decode-step) slots spent
    computing a forward pass for a row that had ALREADY hit its own EOS -
    pure head-of-line blocking cost, paid every step until the slowest row
    in the batch finishes.
"""

import argparse
import time
from typing import Dict, List

import torch

from cpu_utilization import CpuUtilizationSampler
from workload import BATCH_SIZE, MAX_NEW_TOKENS, build_workload, eos_ids_for, load_model


def run_static_batch(model, tokenizer, eos_ids, batch_requests: List[Dict], max_new_tokens: int) -> Dict:
    tokenizer.padding_side = "left"
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in batch_requests
    ]
    encoded = tokenizer(chat_texts, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    batch_size, padded_len = input_ids.shape
    real_prompt_tokens = attention_mask.sum(dim=1).tolist()

    # position_ids account for left-padding: padded (left) positions don't
    # advance the position counter, so the real tokens still see positions
    # 0, 1, 2, ... regardless of how much padding sits to their left.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    batch_start = time.perf_counter()

    with torch.inference_mode():
        out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
    prefill_time_s = time.perf_counter() - batch_start
    past_key_values = out.past_key_values
    next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)  # [batch]
    last_position = position_ids[:, -1:]  # [batch, 1]

    generated: List[List[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    finish_step: List[int] = [None] * batch_size  # step index (1-based) each row naturally finished

    total_row_steps = 0
    wasted_row_steps = 0

    for step in range(max_new_tokens):
        for i in range(batch_size):
            if not finished[i]:
                generated[i].append(next_tokens[i].item())
                if next_tokens[i].item() in eos_ids:
                    finished[i] = True
                    finish_step[i] = step + 1

        if all(finished) or step == max_new_tokens - 1:
            break

        # About to pay for one more batched decode forward pass. Every row
        # rides along regardless of whether it's already finished - tally
        # how many of those batch_size slots are dead weight, right before
        # spending the compute, not after.
        total_row_steps += batch_size
        wasted_row_steps += sum(1 for i in range(batch_size) if finished[i])

        # Finished rows still get fed a real token id (batch tensor shapes
        # are shared) - feeding eos/pad here is harmless since we never
        # record their output past finish_step.
        cur_tokens = next_tokens.unsqueeze(-1)  # [batch, 1]
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype)], dim=1
        )
        last_position = last_position + 1
        with torch.inference_mode():
            out = model(
                cur_tokens, past_key_values=past_key_values,
                attention_mask=attention_mask, position_ids=last_position, use_cache=True,
            )
        past_key_values = out.past_key_values
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

    batch_time_s = time.perf_counter() - batch_start

    padding_tokens = batch_size * padded_len - sum(real_prompt_tokens)
    padding_ratio = padding_tokens / (batch_size * padded_len)
    decode_waste_ratio = wasted_row_steps / total_row_steps if total_row_steps else 0.0

    results = []
    for i, req in enumerate(batch_requests):
        text = tokenizer.decode(generated[i], skip_special_tokens=True)
        results.append({
            "request_id": req["request_id"],
            "category": req["category"],
            "prompt_tokens": real_prompt_tokens[i],
            "output_tokens": len(generated[i]),
            # every row in THIS batch shares the same batch_time_s - nobody
            # gets their result before the whole batch is done. Turned into
            # a cumulative, end-to-end latency_s by the caller, which also
            # adds however long any earlier batches took first.
            "own_batch_time_s": batch_time_s,
            "natural_finish_step": finish_step[i],
            "text": text,
        })

    return {
        "batch_size": batch_size,
        "padded_prompt_len": padded_len,
        "prefill_time_s": prefill_time_s,
        "batch_time_s": batch_time_s,
        "padding_ratio": padding_ratio,
        "decode_waste_ratio": decode_waste_ratio,
        "results": results,
    }


def run_static_batching(max_new_tokens: int = MAX_NEW_TOKENS, batch_size: int = BATCH_SIZE) -> Dict:
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)
    workload = build_workload()
    batches = [workload[i:i + batch_size] for i in range(0, len(workload), batch_size)]

    sampler = CpuUtilizationSampler(interval_s=0.5)
    sampler.start()
    wall_start = time.perf_counter()

    batch_summaries = []
    all_results = []
    cumulative_offset_s = 0.0
    for batch_requests in batches:
        # wait_for_batch_fill_s is 0 here because this fixed workload has
        # every request arrive simultaneously and divides evenly into
        # full batches. In a live system with staggered arrivals, this
        # would be a real additional cost on top of batch_time_s - see README.
        summary = run_static_batch(model, tokenizer, eos_ids, batch_requests, max_new_tokens)
        batch_summaries.append(summary)
        for r in summary["results"]:
            # end-to-end latency = every earlier batch's full duration,
            # since batches run strictly one after another on one CPU,
            # plus this batch's own duration.
            r["latency_s"] = cumulative_offset_s + r["own_batch_time_s"]
        all_results.extend(summary["results"])
        cumulative_offset_s += summary["batch_time_s"]

    total_time_s = time.perf_counter() - wall_start
    cpu_samples = sampler.stop()

    total_output_tokens = sum(r["output_tokens"] for r in all_results)

    return {
        "approach": "static_batching",
        "batch_size": batch_size,
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "avg_cpu_percent": CpuUtilizationSampler.average(cpu_samples),
        "cpu_samples": cpu_samples,
        "avg_padding_ratio": sum(b["padding_ratio"] for b in batch_summaries) / len(batch_summaries),
        "avg_decode_waste_ratio": sum(b["decode_waste_ratio"] for b in batch_summaries) / len(batch_summaries),
        "batch_summaries": batch_summaries,
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("[static_batching] loading model + workload ...", flush=True)
    summary = run_static_batching(args.max_new_tokens, args.batch_size)

    print(f"[static_batching] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[static_batching] throughput_tok_s = {summary['throughput_tok_s']:.2f}")
    print(f"[static_batching] avg_cpu_percent = {summary['avg_cpu_percent']:.1f}%")
    print(f"[static_batching] avg_padding_ratio = {summary['avg_padding_ratio']:.1%}  "
          f"(compute spent on left-padding, not real prompt tokens)")
    print(f"[static_batching] avg_decode_waste_ratio = {summary['avg_decode_waste_ratio']:.1%}  "
          f"(row-steps spent on already-finished rows, held hostage by the batch)")

    short_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "short"]
    long_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "long"]
    print(f"[static_batching] short-request avg latency = {sum(short_latencies)/len(short_latencies):.2f}s "
          f"(this is the head-of-line blocking cost - a short request waits for the WHOLE batch)")
    print(f"[static_batching] long-request avg latency  = {sum(long_latencies)/len(long_latencies):.2f}s")

    for r in summary["results"]:
        print(f"  [{r['request_id']:2d}] {r['category']:5s} "
              f"prompt_tokens={r['prompt_tokens']:3d} output_tokens={r['output_tokens']:3d} "
              f"natural_finish_step={r['natural_finish_step']} latency={r['latency_s']:.3f}s")


if __name__ == "__main__":
    main()
