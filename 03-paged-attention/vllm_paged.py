"""
Module 03 - Approach 3: vLLM as the reference point.

Fires the exact same 24-request workload (from kv_cache.py - same prompts,
same order, same shared MEMORY_BUDGET_BYTES) at vLLM's offline LLM() API,
configured to use the SAME KV-cache memory budget as the other two
approaches, and lets vLLM's real PagedAttention block manager decide
admission/growth/eviction itself. No hand-rolled free list, no hand-rolled
block table - this is the real implementation manual_paged_attention.py is
a simplified, honest stand-in for (see that module's docstring, and this
module's README "Gotcha").

vLLM's CPU backend sizes its KV cache via the VLLM_CPU_KVCACHE_SPACE
environment variable (GiB) rather than a GPU-memory-utilization fraction,
since there's no CUDA allocator to size a fraction of - so that's set here
to MEMORY_BUDGET_BYTES before vLLM is imported, to keep the budget
genuinely identical across all three scripts.

ENVIRONMENT NOTE (same as modules 01 and 02): vLLM ships no Windows
wheels. This script honestly reports "vLLM unavailable" rather than
fabricating numbers whenever `import vllm` fails. If you're running this
from Linux or WSL2 with `pip install vllm` done, it exercises the real
vLLM path and reports vLLM's own block-manager accounting
(cache_config.num_gpu_blocks / block_size) alongside the benchmark numbers.
"""

import argparse
import os
import time
from typing import Dict

from kv_cache import MAX_NEW_TOKENS, MEMORY_BUDGET_BYTES, MODEL_NAME, build_workload

BUDGET_GIB = MEMORY_BUDGET_BYTES / (1024 ** 3)
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", str(BUDGET_GIB))


def check_vllm_available() -> "tuple[bool, str]":
    import platform
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        reason = (
            f"vLLM is not importable in this environment ({e}). "
            f"Running on {platform.system()} {platform.release()}. "
            "Same limitation established in modules 01 and 02: vLLM ships no "
            "Windows wheels and this machine has no usable Linux/WSL2 distro."
        )
        return False, reason
    return True, ""


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_paged(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    workload = build_workload()
    chat_texts = [build_chat_text(tokenizer, r["prompt"]) for r in workload]

    t0 = time.perf_counter()
    # enforce_eager=True: CPU backend has no CUDA graphs to capture.
    llm = LLM(model=MODEL_NAME, dtype="float32", enforce_eager=True)
    load_time_s = time.perf_counter() - t0

    cache_config = getattr(llm.llm_engine, "cache_config", None)
    block_size_tokens = getattr(cache_config, "block_size", None)
    total_blocks = getattr(cache_config, "num_gpu_blocks", None) or getattr(cache_config, "num_cpu_blocks", None)

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
            "latency_s": finish_s if finish_s is not None else total_time_s,
            "text": output.outputs[0].text,
        })

    return {
        "approach": "vllm_paged",
        "budget_gib": BUDGET_GIB,
        "block_size_tokens": block_size_tokens,
        "total_blocks": total_blocks,
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
        print("[vllm_paged] SKIPPED - vLLM cannot run in this environment.")
        print(f"[vllm_paged] reason: {reason}")
        print("[vllm_paged] no numbers are reported below - honest result, not a bug.")
        return

    print(f"[vllm_paged] loading {MODEL_NAME} via vLLM LLM() with "
          f"VLLM_CPU_KVCACHE_SPACE={BUDGET_GIB:.4f} GiB ...", flush=True)
    summary = run_vllm_paged(args.max_new_tokens)
    print(f"[vllm_paged] model loaded in {summary['load_time_s']:.2f}s")
    print(f"[vllm_paged] vLLM block_size_tokens = {summary['block_size_tokens']}")
    print(f"[vllm_paged] vLLM total_blocks = {summary['total_blocks']}")
    print(f"[vllm_paged] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[vllm_paged] throughput_tok_s = {summary['throughput_tok_s']:.2f}")


if __name__ == "__main__":
    main()
