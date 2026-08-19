# 01 - Prefill vs. Decode

Module 00 already split prefill and decode by hand instead of calling
`generate()`, but only showed one version of decode. This module isolates
the two phases on their own (no server, no concurrency) and adds the
comparison that actually proves prefill and decode are two *different kinds*
of computation, not just two timers wrapped around the same loop:

1. **Prefill is one parallel pass over the whole prompt.** Cost scales with
   prompt length. It doesn't matter whether you keep the resulting KV cache
   afterward - the forward pass itself costs the same either way.
2. **Decode *with* a KV cache is memory-bound and (roughly) flat per step.**
   Each step forward-passes exactly one new token and reuses everything
   computed before it via `past_key_values`.
3. **Decode *without* a KV cache degrades into a chain of prefills.** Each
   step re-runs the full forward pass over the entire sequence generated so
   far. Cost per step tracks total sequence length, the same way prefill's
   cost does - because that's what it actually is.
4. **vLLM** is included as a third, external reference point for the same
   prompt/token count - see the environment note below for why it doesn't
   run on this particular machine.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Windows: `WinError 206` / "filename or extension is too long"?** Same
> issue as module 00 - torch ships deeply nested license files and Windows
> caps full paths at 260 characters. If your `.venv` lives somewhere with a
> long path (this repo's `Core Platform` folder plus a nested venv path hit
> exactly this), either enable long paths once, system-wide, in an **admin**
> PowerShell:
> ```powershell
> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
> ```
> or put the venv somewhere short instead, e.g. `python -m venv C:\mi-venv-01`,
> then activate and install from there.

### vLLM (third comparison point)

`pip install vllm` is deliberately **not** part of `requirements.txt`.
**vLLM ships no Windows wheels** - checked against PyPI directly
(`pip index versions vllm` and `pip download --platform win_amd64` for every
release through 0.27.1: manylinux-only binaries, zero `win_amd64` files). It
requires native Linux, or Windows via WSL2 with a real Linux distro
installed inside it (not just Docker Desktop's internal WSL utility VMs,
which is all this machine had available).

Both `vllm_prefill_decode.py` and `benchmark.py` detect this at runtime
(`import vllm` failing) and **report it clearly instead of faking numbers**.
If you're running this module from Linux or WSL2 with a real distro, install
vLLM separately (`pip install vllm`) and the same scripts will exercise the
real vLLM path.

## Run

```bash
python manual_prefill_decode.py                 # one prompt, both manual variants, side by side
python vllm_prefill_decode.py                    # vLLM reference point (or a clear skip message)
python benchmark.py                              # short vs. long prompt, all approaches, comparison table
```

All three accept `--max-new-tokens` (default 20); `manual_prefill_decode.py`
and `vllm_prefill_decode.py` also accept `--prompt`.

## What manual_prefill_decode.py measures

Both variants share the same **PREFILL**: `model(input_ids, use_cache=...)`
once, over the full prompt, timed as a single number. Then they diverge on
**DECODE**:

- `run_with_cache`: loop of `model(one_new_token, past_key_values=..., use_cache=True)`.
  Every step feeds exactly one token; all prior context lives in the cache.
- `run_no_cache`: loop of `model(full_sequence_so_far, use_cache=False)`.
  Every step re-forward-passes the entire sequence generated so far, from
  scratch, one token longer than the step before it. No cache is ever kept
  or reused.

`len(decode_times_s) == output_tokens - 1` for both, same reason as module
00: the first output token is a free byproduct of the prefill pass's logits,
so the decode loop only runs for every token after that.

## What benchmark.py measures

Runs both manual variants (and vLLM, if importable) on a **short (~65
token)** and a **long (~480 token)** prompt, same `max_new_tokens` both
times, and prints one row per (prompt length, approach):

| Column | Meaning |
|---|---|
| `Prefill(s)` | the single prefill forward-pass time |
| `AvgDecode/tok(s)` | mean of every decode step |
| `FirstDecode(s)` / `LastDecode(s)` | first vs. last decode step - the interesting number for `no_cache`, where these should visibly differ |

Expected shape of the results:

- `Prefill(s)` grows a lot from the short row to the long row, for **both**
  manual variants - it's one pass over every prompt token, so it scales
  with prompt length regardless of caching.
- `with_cache` `AvgDecode/tok(s)` stays roughly **flat** whether the prompt
  is short or long - each step is still just one new token plus a cache
  lookup.
- `no_cache` `AvgDecode/tok(s)` is dramatically worse than `with_cache` at
  the **same** prompt length, and gets much worse again on the long prompt -
  it's redoing prefill-sized work every single step.
- The vLLM section either reports `vllm_total_time_s` (and `vllm_ttft_s` if
  the installed version's `RequestOutput.metrics` exposes
  `first_token_time`/`arrival_time`) for both prompt lengths, or explains
  clearly why it couldn't run. vLLM's own internal prefill/decode split
  isn't traced here - that's module 07.

If `with_cache` and `no_cache` decode times look similar at a given prompt
length, that's a smell: check that `run_no_cache` is actually being called
with `use_cache=False` and the growing `full_seq`, not accidentally reusing
a cache.
