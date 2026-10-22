# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real >2**31 QKV element-offset branch, beyond meta admission."""

import argparse
import datetime
import json
import subprocess
import sys
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
actual_head = subprocess.check_output(
    ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
).strip()
if actual_head != "9027363e6641fb211526c20ae005251eb3d2f45d":
    parser.error("Checkout does not match tested PR head")
sys.path.insert(0, str(args.checkout.resolve()))
import einops
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.fused_vision_qk_rope import (
    _supported_layout,
    try_fused_vision_qk_rope,
)

output = args.output
output.mkdir(parents=True, exist_ok=False)
record = dict(
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    seq=400000,
    batch=2,
    dtype="torch.bfloat16",
    seed=125,
    rtol=0,
    atol=0,
    passed=False,
    note="Supplemental synthetic kernel-only boundary test; no model weights",
)
try:
    torch.set_grad_enabled(False)
    g = torch.Generator(device="cuda").manual_seed(125)
    x = torch.randn((400000, 2, 3456), device="cuda", dtype=torch.bfloat16, generator=g)
    original = x.clone()
    angle = torch.randn((400000, 36), device="cuda", generator=g)
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    assert x.numel() > 2**31 and _supported_layout(x, cos, sin, 16, 72)
    record.update(
        x_shape=list(x.shape),
        x_stride=list(x.stride()),
        x_numel=x.numel(),
        cos_shape=list(cos.shape),
        cos_stride=list(cos.stride()),
        sin_stride=list(sin.stride()),
    )
    with set_current_vllm_config(VllmConfig()):
        rotary = ApplyRotaryEmb(enforce_enable=True)
        assert rotary._forward_method.__name__ == "forward_cuda"
        actual = try_fused_vision_qk_rope(
            x, cos, sin, 16, 72, is_neox_style=True, enable_fp32_compute=False
        )
        assert actual is not None
        qkv = einops.rearrange(x, "s b (three h d) -> b s three h d", three=3, h=16)
        qk = einops.rearrange(qkv[:, :, :2], "b s two h d -> (two b) s h d", two=2)
        expected = rotary(qk.contiguous(), cos, sin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(x, original)
        torch.cuda.synchronize()
    record.update(
        passed=True,
        input_unchanged=True,
        comparison="full tensor exact",
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
except Exception as e:
    record["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()
finally:
    record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
sys.exit(0 if record["passed"] else 1)
