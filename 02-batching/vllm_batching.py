"""
Module 02 - Approach 4: vLLM as the reference point.

Fires the exact same 16-request workload (from workload.py - same prompts,
same order, same shared MAX_NEW_TOKENS cap) at vLLM's offline LLM() API in
one batch generate() call, and lets vLLM's real continuous-batching
scheduler decide admission/eviction/ordering itself. No hand-rolled
padding, no hand-rolled cache surgery - this is what the toy scheduler in
continuous_batching_toy.py is a simplified stand-in for.

ENVIRONMENT NOTE (same as module 01's vllm_prefill_decode.py): vLLM ships
no Windows wheels. This repo's benchmark honestly reports "vLLM
unavailable" rather than fabricating numbers whenever `import vllm` fails.
If you're running this from Linux or WSL2 with `pip install vllm` done,
this script executes the real vLLM path.
"""

import argparse
import time
from typing import Dict

from workload import MAX_NEW_TOKENS, MODEL_NAME, build_workload


def check_vllm_available() -> "tuple[bool, str]":
    import platform
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        reason = (
            f"vLLM is not importable in this environment ({e}). "
            f"Running on {platform.system()} {platform.release()}. "
            "Same limitation established in module 01: vLLM ships no Windows "
            "wheels and this machine has no usable Linux/WSL2 distro."
        )
        return False, reason
    return True, ""


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_batching(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    workload = build_workload()
    chat_texts = [build_chat_text(tokenizer, r["prompt"]) for r in workload]

    t0 = time.perf_counter()
    # enforce_eager=True: CPU backend has no CUDA graphs to capture.
    llm = LLM(model=MODEL_NAME, dtype="float32", enforce_eager=True)
    load_time_s = time.perf_counter() - t0

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    wall_start = time.perf_counter()
    outputs = llm.generate(chat_texts, sampling_params)
    total_time_s = time.perf_counter() - wall_start

    results = []
    total_output_tokens = 0
    for req, output in zip(workload, outputs):
        output_tokens = len(output.outputs[0].token_ids)
        total_output_tokens += output_tokens

        finish_s = None
        metrics = getattr(output, "metrics", None)
        if metrics is not None:
            arrival = getattr(metrics, "arrival_time", None)
            finished = getattr(metrics, "finished_time", None)
            if arrival is not None and finished is not None:
                finish_s = finished - arrival

        results.append({
            "request_id": req["request_id"],
            "category": req["category"],
            "output_tokens": output_tokens,
            # per-request completion time if vLLM's metrics expose it;
            # otherwise every request in the batch shares the same
            # end-to-end total_time_s, same convention as the other
            # approaches when a finer number isn't available.
            "latency_s": finish_s if finish_s is not None else total_time_s,
            "text": output.outputs[0].text,
        })

    return {
        "approach": "vllm_batching",
        "load_time_s": load_time_s,
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    available, reason = check_vllm_available()
    if not available:
        print("[vllm_batching] SKIPPED - vLLM cannot run in this environment.")
        print(f"[vllm_batching] reason: {reason}")
        print("[vllm_batching] no numbers are reported below - honest result, not a bug.")
        return

    print(f"[vllm_batching] loading {MODEL_NAME} via vLLM LLM() ...", flush=True)
    summary = run_vllm_batching(args.max_new_tokens)
    print(f"[vllm_batching] model loaded in {summary['load_time_s']:.2f}s")
    print(f"[vllm_batching] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[vllm_batching] throughput_tok_s = {summary['throughput_tok_s']:.2f}")

    short_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "short"]
    long_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "long"]
    print(f"[vllm_batching] short-request avg latency = {sum(short_latencies)/len(short_latencies):.2f}s")
    print(f"[vllm_batching] long-request avg latency  = {sum(long_latencies)/len(long_latencies):.2f}s")


if __name__ == "__main__":
    main()
