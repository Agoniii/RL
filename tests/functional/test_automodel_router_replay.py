# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiny real-model R3 tests requiring CUDA; no vLLM or distributed workers."""

import pytest
import torch
from nemo_automodel.components.moe.layers import Gate
from torch.utils.checkpoint import checkpoint

from nemo_rl.experience.r3 import RouteSource
from nemo_rl.models.automodel.router_replay import (
    automodel_router_replay_context,
    verify_automodel_router_replay,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The pinned Qwen3 MoE constructor requires CUDA",
)


@pytest.mark.parametrize("ac", [False, True])
def test_tiny_qwen3_model_replay_padding_and_backward(ac):
    """Exercise real batch-major MoE flattening, attention and backward on CUDA."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM
    from transformers import Qwen3MoeConfig

    torch.manual_seed(44)
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=32,
        dtype="float32",
        tie_word_embeddings=False,
    )
    with torch.device("cuda"):
        model = Qwen3MoeForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                experts="torch",
                dispatcher="torch",
                rope_fusion=False,
            ),
            moe_overrides={
                "enable_routing_replay": True,
                "aux_loss_coeff": 0.0,
                "gate_bias_update_factor": 0.0,
            },
        )
    model.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
    captured = []
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[1].detach().clone())
        )
        for module in model.modules()
        if isinstance(module, Gate)
    ]
    with torch.no_grad():
        native_logits = model(ids).logits
    for hook in hooks:
        hook.remove()
    routes = torch.stack(captured, dim=1).reshape(2, 4, 2, 2)
    source = torch.full(
        (2, 4, 2), int(RouteSource.CAPTURED), dtype=torch.int8, device="cuda"
    )
    source[:, -1] = int(RouteSource.TERMINAL_UNUSED)

    def gradients(padding):
        model.zero_grad(set_to_none=True)
        tokens = torch.nn.functional.pad(ids, (0, padding))
        targets = torch.nn.functional.pad(routes, (0, 0, 0, 0, 0, padding), value=-1)
        sources = torch.nn.functional.pad(
            source, (0, 0, 0, padding), value=int(RouteSource.PADDING)
        )

        def forward(value):
            return model(
                value, attention_mask=torch.ones_like(value, dtype=torch.bool)
            ).logits

        with automodel_router_replay_context(
            model, targets, sources, enabled=True
        ) as assignments:
            logits = (
                checkpoint(forward, tokens, use_reentrant=False)
                if ac
                else forward(tokens)
            )
            torch.testing.assert_close(
                logits[:, :3], native_logits[:, :3], atol=2e-6, rtol=1e-5
            )
            loss = torch.nn.functional.cross_entropy(
                logits[:, :3].reshape(-1, 32), ids[:, 1:].reshape(-1)
            )
            loss.backward()
            verify_automodel_router_replay(assignments)
        # A repeated dummy forward must not inject gradients from auxiliary loss.
        with automodel_router_replay_context(model, targets, sources, enabled=True):
            (forward(tokens).sum() * 0).backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    baseline, padded = gradients(0), gradients(3)
    assert baseline.keys() == padded.keys()
    for name in baseline:
        torch.testing.assert_close(padded[name], baseline[name], atol=3e-6, rtol=1e-4)
    assert any(
        "gate.weight" in name and grad.abs().sum() > 0
        for name, grad in baseline.items()
    )
