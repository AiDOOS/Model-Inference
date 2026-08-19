# Results

## Environment

- CPU-only, 12GB RAM, Windows 10.
- Qwen/Qwen2.5-0.5B-Instruct, float32, already cached locally from module 00.
- vLLM: **not runnable on this machine.** Confirmed via `pip index versions
  vllm` and `pip download vllm --platform win_amd64 --only-binary=:all:`
  against every release through 0.27.1 - zero Windows wheels exist. This
  machine has no general-purpose WSL Linux distro installed (only Docker
  Desktop's internal WSL utility VMs), so there's no Linux environment here
  to install vLLM into either. Both `vllm_prefill_decode.py` and
  `benchmark.py` detect and report this at runtime rather than fabricating
  numbers - see the run below.

## manual_prefill_decode.py (2026-08-19)

```
python manual_prefill_decode.py --max-new-tokens 10
```

Default prompt, `prompt_tokens=65`, `max_new_tokens=10` (9 decode steps).

```
prefill_time_s (with_cache) : 1.4460s
prefill_time_s (no_cache)   : 1.0764s

decode_times_s (with_cache) : n=9  mean=0.1748s  first=0.1742s  last=0.1947s  min=0.1689s  max=0.1947s
decode_times_s (no_cache)   : n=9  mean=1.0976s  first=1.0458s  last=1.1401s  min=1.0454s  max=1.1614s
```

**Reading it:**

- `prefill_time_s` is ~equal between the two variants (1.45s vs 1.08s, same
  ballpark) - it's the same forward pass either way, whether or not the
  cache gets kept afterward.
- `with_cache` decode is small and flat: ~0.17-0.19s per step regardless of
  position.
- `no_cache` decode is 6x worse per step (~1.05-1.14s) and is itself in the
  same ballpark as `prefill_time_s` - because at that point it basically
  *is* a prefill, over a slightly longer sequence each time.

## benchmark.py (2026-08-19)

```
python benchmark.py --max-new-tokens 10
```

Reduced run (10 tokens instead of the 20 default) used to verify the script
end-to-end and get a first real reading; the shape of the result doesn't
depend on token count.

```
 PromptLen        Approach   Prefill(s)   Steps   AvgDecode/tok(s)   FirstDecode(s)   LastDecode(s)
---------------------------------------------------------------------------------------------------
        65      with_cache       1.1374       9             0.1763           0.1871          0.1794
        65        no_cache       0.9634       9             0.9658           0.9663          1.0221
       482      with_cache       4.3501       9             0.1899           0.2310          0.1795
       482        no_cache       4.0861       9             4.0617           4.1117          4.1214

vLLM reference:
  N/A - vLLM is not importable in this environment (No module named 'vllm'). Running
  on Windows 10. vLLM ships no Windows wheels (checked PyPI: manylinux-only through
  0.27.1) - it requires native Linux or WSL2 with a real Linux distro. This machine
  has no such distro installed (only Docker Desktop's internal WSL VMs).
```

**Reading it:**

- **Prefill scales with prompt length**, for both variants: 65 tokens ->
  ~1.1-1.4s, 482 tokens (7.4x more prompt tokens) -> ~4.1-4.4s (~3.8x).
  Sub-linear rather than proportional at this scale - a 0.5B model on CPU
  has enough fixed dispatch/overhead cost that it doesn't disappear even at
  65 tokens - but the direction and the fact that it moves at all is the
  point: prefill cost is driven by prompt length.
- **`with_cache` decode/token barely moves**: ~0.176-0.190s whether the
  prompt was 65 or 482 tokens. Memory-bound, one new token per step,
  ~independent of how much context is behind it (at these context lengths).
- **`no_cache` decode/token tracks prefill almost exactly** at both prompt
  lengths (0.96s vs. 1.14s prefill at 65 tokens; 4.06-4.12s vs. 4.35s
  prefill at 482 tokens) - direct confirmation that a decode step without a
  cache *is* a prefill, computationally, just over a sequence that's grown
  by a handful of tokens.

## vLLM (2026-08-19)

```
python vllm_prefill_decode.py
```

```
[vllm] SKIPPED - vLLM cannot run in this environment.
[vllm] reason: vLLM is not importable in this environment (No module named 'vllm').
Running on Windows 10. vLLM ships no Windows wheels (checked PyPI: manylinux-only
through 0.27.1) - it requires native Linux or WSL2 with a real Linux distro. This
machine has no such distro installed (only Docker Desktop's internal WSL VMs).
[vllm] no numbers are reported below - this is the honest result for this module,
not a bug. See README.md for details.
```

No fabricated vLLM numbers exist anywhere in this module. If this repo is
ever run from Linux or a real WSL2 distro, `pip install vllm` followed by
the same two commands will produce real `vllm_total_time_s` (and
`vllm_ttft_s`, version permitting) numbers without any code changes.

## Full benchmark (20 tokens, default)

Not yet run - the 10-token reduced run above already shows the expected
shape clearly. Run `python benchmark.py` (default `--max-new-tokens 20`) and
append results here if a longer, noisier-averaged confirmation is wanted.
