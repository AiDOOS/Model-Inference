"""
Module 01 - vLLM as a third reference point.

Same prompt, same token count, same model as manual_prefill_decode.py, but
served through vLLM's offline LLM() API instead of hand-rolled
model(...) calls. vLLM does its own prefill (parallel pass over the prompt)
and decode (sequential, cached, batched-under-the-hood) internally - this
script does NOT reach into vLLM's scheduler to separate those phases. It
reports one end-to-end number as a reference point against the two manual
approaches, plus time-to-first-token when vLLM's own RequestOutput.metrics
exposes it (varies by version). A real prefill/decode trace *inside* vLLM's
engine is module 07's job, not this one's.

IMPORTANT - environment note for this repo:
vLLM publishes NO Windows wheels (verified against PyPI: every release up
to 0.27.1 ships manylinux-only binaries, zero win_amd64 files). It only
runs on Linux, or on Windows via WSL2 with a real Linux distro installed
inside it. This machine is native Windows with no general-purpose WSL
distro present (only Docker Desktop's internal WSL utility VMs, which
aren't a place to pip install into). So on THIS machine, `import vllm`
below fails on purpose, and this script reports that clearly instead of
fabricating numbers. If you run this same file later from Linux/WSL2 with
`pip install vllm` done, it will actually execute the vLLM path.
"""

import argparse
import platform
import time
from typing import Dict, Optional

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_PROMPT = (
    "Explain, in a few sentences, why the sky appears blue during the day "
    "and orange or red at sunset. Keep the explanation accessible to a "
    "curious 10-year-old."
)


def check_vllm_available() -> "tuple[bool, str]":
    """Returns (available, reason). Never raises."""
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        reason = (
            f"vLLM is not importable in this environment ({e}). "
            f"Running on {platform.system()} {platform.release()}. "
            "vLLM ships no Windows wheels (checked PyPI: manylinux-only through "
            "0.27.1) - it requires native Linux or WSL2 with a real Linux distro. "
            "This machine has no such distro installed (only Docker Desktop's "
            "internal WSL VMs)."
        )
        return False, reason
    return True, ""


def build_chat_text(prompt: str) -> str:
    """Apply the exact same chat template manual_prefill_decode.py uses, so
    prompt_tokens matches between the manual and vLLM runs for a fair
    comparison. Requires transformers (already a dependency of this repo)
    purely for tokenizer/template access - vLLM itself does its own
    tokenization internally at generate() time."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer(chat_text).input_ids)
    return chat_text, prompt_tokens


def run_vllm(prompt: str, max_new_tokens: int) -> Dict:
    """Actual vLLM path. Only reachable if `import vllm` succeeds."""
    from vllm import LLM, SamplingParams

    chat_text, prompt_tokens = build_chat_text(prompt)

    t0 = time.perf_counter()
    # enforce_eager=True: CPU backend has no CUDA graphs to capture anyway;
    # this avoids vLLM wasting time attempting graph capture on CPU.
    llm = LLM(model=MODEL_NAME, dtype="float32", enforce_eager=True)
    load_time_s = time.perf_counter() - t0

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate([chat_text], sampling_params)
    vllm_total_time_s = time.perf_counter() - t0

    output = outputs[0]
    output_tokens = len(output.outputs[0].token_ids)
    text = output.outputs[0].text

    vllm_ttft_s: Optional[float] = None
    metrics = getattr(output, "metrics", None)
    if metrics is not None:
        arrival = getattr(metrics, "arrival_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        if arrival is not None and first_token is not None:
            vllm_ttft_s = first_token - arrival

    return {
        "vllm_total_time_s": vllm_total_time_s,
        "vllm_ttft_s": vllm_ttft_s,
        "load_time_s": load_time_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    available, reason = check_vllm_available()
    if not available:
        print("[vllm] SKIPPED - vLLM cannot run in this environment.")
        print(f"[vllm] reason: {reason}")
        print(
            "[vllm] no numbers are reported below - this is the honest result "
            "for this module, not a bug. See README.md for details."
        )
        return

    print(f"[vllm] loading {MODEL_NAME} via vLLM LLM() ...", flush=True)
    result = run_vllm(args.prompt, args.max_new_tokens)
    print(f"[vllm] model loaded in {result['load_time_s']:.2f}s")
    print(f"[vllm] prompt_tokens={result['prompt_tokens']}  output_tokens={result['output_tokens']}")
    print(f"[vllm] vllm_total_time_s = {result['vllm_total_time_s']:.4f}s")
    if result["vllm_ttft_s"] is not None:
        print(f"[vllm] vllm_ttft_s       = {result['vllm_ttft_s']:.4f}s")
        remaining = result["vllm_total_time_s"] - result["vllm_ttft_s"]
        print(f"[vllm] remaining decode time = {remaining:.4f}s "
              f"over {result['output_tokens'] - 1} tokens")
    else:
        print(
            "[vllm] vllm_ttft_s not available from this vLLM version's "
            "RequestOutput.metrics - reporting vllm_total_time_s only. A real "
            "prefill/decode trace inside vLLM's engine is module 07's job."
        )


if __name__ == "__main__":
    main()
