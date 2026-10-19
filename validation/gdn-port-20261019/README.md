# GDN inverse port validation

Candidate: `be22e9878d7c47459bc860ac771895d4d610d4cf`.

The solver, tests and kernel benchmark are on branch
`c550/port-high-impact-20261019`. This evidence branch is separate from the
proposed code diff.

- `kernel-results.json`: 13 C550 tests, including changed-length graph replay,
  and five alternating eager event-timing pairs per workload.
- `model-results.json`: 8 real Qwen3.5-35B-A3B W8A8 requests, exact greedy
  token-sequence parity; the candidate solver's eligible path executed 210 times.
- No complete current-upstream runtime, NVIDIA/ROCm result, official model
  accuracy score or whole-model throughput improvement is claimed.

Kernel reproduction in a compatible installed upstream environment:

```bash
.venv/bin/python -m pytest tests/kernels/fla/test_solve_tril.py -v
.venv/bin/python benchmarks/kernels/benchmark_gdn_solve_tril.py --timing event
```

The model scripts are parameterized copies of the locally tested scripts;
only fixed paths and package selection were parameterized and unused local
RPC function definitions removed. They require a compatible installed engine
with `LLM.collective_rpc` and worker extensions. The recorded execution used
MetaX Torch 2.10 and the older pinned MetaX FLA dependencies, with the exact
new solver source loaded at runtime. They are not a full-upstream-build test.

Set `GDN_MODEL_PATH` to the local checkpoint, `GDN_SOLVER_SOURCE` to the new
solver file, and `GDN_FLA_PACKAGE` to the installed engine's FLA ops package
(`vllm.model_executor.layers.fla.ops` for the tested engine, or
`vllm.third_party.flash_linear_attention.ops` for a compatible current engine).
Run the script from this directory with this directory on `PYTHONPATH`, using
the compatible engine's virtual-environment Python. CUDA_VISIBLE_DEVICES must
select an otherwise idle GPU with room for the model. Vendor-specific runtime
variables and dependencies must already be configured. Both variants use
TP1, eager mode, no MTP, no prefix cache, temperature 0, seed 20261008, output
limit 128, context 8192 and two active sequences.

These are functional regression cases, not a statistically powered accuracy
benchmark. Inputs are self-contained synthetic prompts; no production requests,
credentials, model weights or raw server logs are included.
