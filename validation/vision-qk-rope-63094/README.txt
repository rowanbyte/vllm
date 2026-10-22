PR #63094 NVIDIA kernel validation
=================================
PR: https://github.com/vllm-project/vllm/pull/63094
Tested head: 9027363e6641fb211526c20ae005251eb3d2f45d
Compared base: b3b0bc30590748b859af89832ccc50482dc9b08e
This appendix is on an independent evidence branch. It does not change the PR
implementation. results.json includes all 24 cases, both modes, statistics,
profiling summaries, correctness and limitations; raw_run_1/2.json retain every
timing sample and arm order. No outliers were removed. summary.csv has all 48
case/mode rows, including FP16 fallback.

Environment and setup
---------------------
B300 SXM6 AC, SM10.3; Python 3.12.3; torch 2.14.0+cu130; CUDA 13.0;
Triton 3.8.0; cuDNN 9.24; driver 580.82.07; nvcc 13.0.88.
Only kernel dependencies were validated. The inherited environment had unrelated
serving dependency conflicts; this is not a complete serving installation.
The native extension and flash-attention rotary assets were byte-matched to the
official cu130 wheel for the exact base commit. Wheel SHA256:
cf7256de60b000d85fdb2507c904a02be52f4d5c47da8bb34298798e482c1d05
Official wheel index:
https://wheels.vllm.ai/b3b0bc30590748b859af89832ccc50482dc9b08e/cu130
Use a task-local venv with this torch/CUDA/Triton stack and matching vLLM native
extensions (build the pinned source, or use the exact-base official wheel).
Do not overwrite a shared installation. These commands assume that configured
kernel environment; they are not a generic serving dependency installer.

Reproduce (select an allocated GPU; no model weights required)
-------------------------------------------------------------
git clone https://github.com/rowanbyte/vllm.git vllm-pr63094
cd vllm-pr63094
git checkout 9027363e6641fb211526c20ae005251eb3d2f45d
# Save this evidence directory separately as ../evidence.
# Set GPU to an allocated device identifier, e.g. export GPU=0.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m pytest tests/kernels/core/test_fused_vision_qk_rope.py -vv
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python ../evidence/benchmark.py --checkout . --output ../run1 --rounds 15 --inner 64 --graph-replays 3 --order-offset 0
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python ../evidence/benchmark.py --checkout . --output ../run2 --rounds 15 --inner 64 --graph-replays 3 --order-offset 1
# Optional large synthetic boundary check: measured peak ~23.3 GiB allocated;
# allow at least 32 GiB free GPU memory, plus context/reservation headroom.
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python ../evidence/check_wide_index.py --checkout . --output ../wide-index

Both scripts reject a checkout whose HEAD differs from the tested SHA. Their
portable CLI/head checks and metadata sanitization were CPU checked; the original
harness produced the stored GPU measurements. Numerical/timing functions are
AST-identical (see provenance.json). Packaging was not itself rerun on GPU.

Correctness and comparison
--------------------------
11 contributed tests passed, 0 failed/skipped: 2 metadata, 8 exact native-parity
cases (S=17/257/8192/46080, B=1/2, contiguous/strided tables), 1 changed-input
CUDA graph replay. Seed 125, rtol=atol=0, full-tensor comparisons, input unchanged.
Additional GPU check: QKV [400000,2,3456], 2,764,800,000 elements, exact against
native and input unchanged. Each of 24 benchmark cases passed exact comparison
in both independent processes.

Baseline: the pre-PR QKV reshape, contiguous QK packing and real
ApplyRotaryEmb.forward_cuda. Optimized: try_fused_vision_qk_rope plus the same
native fallback if unsupported. Same pristine tensors in each arm.
H=16, D=72, QKV [S,B,3456] contiguous; cos/sin [S,36], strides [36,1] or [72,2].
S/B = 17/1,256/1,257/2,4096/1,8192/1,46080/1; BF16/FP16; 2 table layouts.
Output [2*B,S,16,72]. No model weights, encoder layer or serving execution.

30 warmup calls per arm. Each process has 15 alternating-order rounds, second
process starts in reverse order: 30 samples/arm/case/mode, 15 per starting order.
Eager sample: CUDA events around 64 operations. Graph: capture 64 operations,
5 warmup replays, time 3 replays per sample, normalize to per-operation time.
JIT excluded. No cache flushing. p50/p90 are percentiles of per-operation batch
averages, not single-call tail latencies. Eager timings include host enqueue and
allocation gaps; graph reduces those gaps. Normal operation allocations retained.

Results and boundaries
----------------------
BF16 graph component speedup: 2.87-6.70x; S46080/B1 contiguous: 5.93x.
The latter operation's peak allocated increment is 405 -> 202.5 MiB (50% lower),
excluding already-live inputs, allocator reserved memory and graph pools.
Contiguous BF16 launch counts: 2 -> 1; strided tables: 4 -> 1 in profiled cases.
FP16 remains fallback: graph speedup 0.982-1.002x, eager 0.944-1.000x;
worst eager small case takes about 6% more time. All cases are retained.
Default power/clocks; shared host. Only B300/SM10.3 tested. No claim for other
NVIDIA architectures, full encoder/model, decoder, TTFT or serving throughput.
Historical C550 encoder measurements are documented separately in the PR and
are not NVIDIA results. See results.json for all limitations.

Code license: Apache-2.0, matching vLLM. AI assistance was used in preparation.
