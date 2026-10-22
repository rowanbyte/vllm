# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR-exact native packing/rotation versus fused dispatch on one allocated GPU.

Select one allocated device with CUDA_VISIBLE_DEVICES. All timing is CUDA-event timing. BF16 activates fusion;
FP16 explicitly measures the unchanged native fallback. No weights are loaded.
"""

# The per-case globals are created before each call and deleted after timing.
# ruff: noqa: F821
import argparse
import datetime
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rounds", type=int, default=15)
parser.add_argument("--inner", type=int, default=64)
parser.add_argument("--graph-replays", type=int, default=3)
parser.add_argument("--order-offset", type=int, default=0)
args = parser.parse_args()
expected_head = "9027363e6641fb211526c20ae005251eb3d2f45d"
actual_head = subprocess.check_output(
    ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
).strip()
if actual_head != expected_head:
    parser.error(
        f"Expected {expected_head}, got {actual_head}; results would need separate attribution"
    )
sys.path.insert(0, str(args.checkout.resolve()))
import einops
import numpy as np
import torch
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.fused_vision_qk_rope import (
    try_fused_vision_qk_rope,
)
from vllm.platforms import current_platform

assert Path(vllm.__file__).resolve().is_relative_to(args.checkout.resolve())
assert current_platform.is_cuda() and torch.cuda.device_count() == 1
args.output.mkdir(parents=True, exist_ok=False)
torch.set_grad_enabled(False)


def summary(values):
    return dict(
        n=len(values),
        median_us=statistics.median(values),
        p50_us=float(np.percentile(values, 50)),
        p90_us=float(np.percentile(values, 90)),
        mean_us=statistics.mean(values),
        stddev_us=statistics.pstdev(values),
        variance_us2=statistics.pvariance(values),
        cv=statistics.pstdev(values) / statistics.mean(values),
        min_us=min(values),
        max_us=max(values),
    )


def measure(fn, inner):
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / inner


def mem_peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    result_bytes = result.numel() * result.element_size()
    return dict(
        peak_allocated_delta_bytes=torch.cuda.max_memory_allocated() - start,
        live_allocated_delta_bytes=torch.cuda.memory_allocated() - start,
        output_bytes=result_bytes,
    )


report = dict(
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    head=subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip(),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    vllm_version=vllm.__version__,
    seed=125,
    rounds=args.rounds,
    inner=args.inner,
    graph_replays=args.graph_replays,
    warmup_calls=30,
    order_offset=args.order_offset,
    timing="CUDA events; eager inner calls; graph captures inner calls, replays graph-replays times per sample",
    cases=[],
)
(args.output / "config.json").write_text(json.dumps(report, indent=2))

with set_current_vllm_config(VllmConfig()):
    rotary = ApplyRotaryEmb(enforce_enable=True)
    assert rotary._forward_method.__name__ == "forward_cuda", str(
        rotary._forward_method
    )
    report["native_dispatch"] = rotary._forward_method.__qualname__
    for seq, batch in [(17, 1), (256, 1), (257, 2), (4096, 1), (8192, 1), (46080, 1)]:
        for dtype in (torch.bfloat16, torch.float16):
            for strided in (False, True):
                gen = torch.Generator(device="cuda").manual_seed(125)
                x = torch.randn(
                    (seq, batch, 3456), dtype=dtype, device="cuda", generator=gen
                )
                angle = torch.randn(
                    (seq, 72 if strided else 36), device="cuda", generator=gen
                )
                cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
                if strided:
                    cos, sin = cos[:, ::2], sin[:, ::2]
                pristine = x.clone()

                def native():
                    qkv = einops.rearrange(
                        x, "s b (three h d) -> b s three h d", three=3, h=16
                    )
                    qk = einops.rearrange(
                        qkv[:, :, :2], "b s two h d -> (two b) s h d", two=2
                    )
                    return rotary(qk.contiguous(), cos, sin)

                def optimized():
                    result = try_fused_vision_qk_rope(
                        x,
                        cos,
                        sin,
                        16,
                        72,
                        is_neox_style=True,
                        enable_fp32_compute=False,
                    )
                    return native() if result is None else result

                fused = try_fused_vision_qk_rope(
                    x, cos, sin, 16, 72, is_neox_style=True, enable_fp32_compute=False
                )
                assert (fused is not None) == (dtype == torch.bfloat16)
                expected, actual = native(), optimized()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(x, pristine)
                case = dict(
                    seq=seq,
                    batch=batch,
                    dtype=str(dtype),
                    strided=strided,
                    x_shape=list(x.shape),
                    x_stride=list(x.stride()),
                    cos_shape=list(cos.shape),
                    cos_stride=list(cos.stride()),
                    sin_stride=list(sin.stride()),
                    fused_active=fused is not None,
                    seed=125,
                    rtol=0,
                    atol=0,
                    correctness="exact",
                    max_abs_diff=float((actual.float() - expected.float()).abs().max()),
                    memory={},
                    modes={},
                )
                del expected, actual, fused
                for _ in range(30):
                    native()
                    optimized()
                torch.cuda.synchronize()
                case["memory"] = {
                    "baseline": mem_peak(native),
                    "optimized": mem_peak(optimized),
                }
                for mode in ("eager", "cuda_graph"):
                    functions = {"baseline": native, "optimized": optimized}
                    graphs = {}
                    if mode == "cuda_graph":
                        for arm, fn in functions.items():
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(args.inner):
                                    y = fn()
                            graphs[arm] = (graph, y)
                        functions = {
                            arm: pair[0].replay for arm, pair in graphs.items()
                        }
                        for fn in functions.values():
                            for _ in range(5):
                                fn()
                        torch.cuda.synchronize()
                        for arm, pair in graphs.items():
                            torch.testing.assert_close(
                                pair[1], native(), rtol=0, atol=0
                            )
                    samples = {"baseline": [], "optimized": []}
                    paired = []
                    for repeat in range(args.rounds):
                        order = (
                            ["baseline", "optimized"]
                            if (repeat + args.order_offset) % 2 == 0
                            else ["optimized", "baseline"]
                        )
                        row = {"round": repeat, "order": order}
                        for arm in order:
                            loops = (
                                args.inner if mode == "eager" else args.graph_replays
                            )
                            us = measure(functions[arm], loops)
                            if mode == "cuda_graph":
                                us /= args.inner
                            samples[arm].append(us)
                            row[arm + "_us"] = us
                        row["speedup"] = row["baseline_us"] / row["optimized_us"]
                        paired.append(row)
                    data = {
                        "raw_samples_us": samples,
                        "paired_rounds": paired,
                        "baseline": summary(samples["baseline"]),
                        "optimized": summary(samples["optimized"]),
                        "paired_speedup": {
                            k.replace("_us2", "").replace("_us", ""): v
                            for k, v in summary([r["speedup"] for r in paired]).items()
                        },
                    }
                    data["speedup"] = (
                        data["baseline"]["median_us"] / data["optimized"]["median_us"]
                    )
                    case["modes"][mode] = data
                    print(
                        json.dumps(
                            {
                                k: case[k]
                                for k in [
                                    "seq",
                                    "batch",
                                    "dtype",
                                    "strided",
                                    "fused_active",
                                ]
                            }
                            | {
                                "mode": mode,
                                "baseline_us": data["baseline"]["median_us"],
                                "optimized_us": data["optimized"]["median_us"],
                                "speedup": data["speedup"],
                            }
                        ),
                        flush=True,
                    )
                    if mode == "cuda_graph":
                        del graph, y, pair, fn
                    del functions, graphs
                assert torch.equal(x, pristine)
                case["input_unchanged"] = True
                report["cases"].append(case)
                (args.output / "results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                del x, cos, sin, pristine, angle
                gc.collect()
                torch.cuda.synchronize()
report["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
