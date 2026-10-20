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

"""CPU adapter tests using AutoModel's real Gate; no distributed model/vLLM."""

from types import SimpleNamespace

import pytest
import torch
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_rl.experience.r3 import RouteSource
from nemo_rl.models.automodel.router_replay import (
    automodel_router_replay_context,
    verify_automodel_router_replay,
)


def model_and_payload():
    cfg = MoEConfig(
        dim=8,
        inter_dim=16,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        norm_topk_prob=True,
        dtype=torch.float32,
        enable_routing_replay=True,
    )
    model = nn.Module()
    model.layers = nn.ModuleList([Gate(cfg)])
    model.config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=1)
    with torch.no_grad():
        model.layers[0].weight.normal_(0, 0.2)
    routes = torch.tensor([[[[0, 1]], [[-1, -1]], [[2, 3]]]], dtype=torch.int16)
    source = torch.tensor(
        [[[RouteSource.CAPTURED], [RouteSource.MISSING], [RouteSource.CAPTURED]]],
        dtype=torch.int8,
    )
    return model, routes, source


@pytest.mark.parametrize("ac", [None, True, False])
def test_real_gate_padding_dummy_and_checkpoint_preserve_gradients(ac):
    torch.manual_seed(42)
    model, routes, source = model_and_payload()
    gate = model.layers[0]
    inputs = torch.randn(3, 8)

    def run(padding, dummy):
        model.zero_grad(set_to_none=True)
        x = torch.cat((inputs, torch.randn(padding, 8))).requires_grad_()
        targets = torch.cat((routes, torch.full((1, padding, 1, 2), -1)), dim=1)
        sources = torch.cat(
            (source, torch.full((1, padding, 1), int(RouteSource.PADDING))), dim=1
        )

        def forward(value):
            weights, indices, _ = gate(
                value, torch.ones(value.shape[0], dtype=torch.bool), None
            )
            # Nonconstant expert contributions: summing normalized weights alone
            # is a constant objective and cannot test router gradients.
            return (weights * (indices.float() + 1)).sum(-1)

        with automodel_router_replay_context(
            model, targets, sources, enabled=True
        ) as assignments:
            output = (
                forward(x) if ac is None else checkpoint(forward, x, use_reentrant=ac)
            )
            output[:3].square().sum().backward()
            verify_automodel_router_replay(assignments)
        if dummy:
            with automodel_router_replay_context(model, targets, sources, enabled=True):
                (forward(x).sum() * 0).backward()
        return gate.weight.grad.detach().clone()

    baseline = run(0, False)
    torch.testing.assert_close(run(4, True), baseline, atol=1e-6, rtol=1e-5)
    assert baseline.abs().sum() > 0


def test_real_gate_replay_matches_live_scores_and_reference_is_native():
    torch.manual_seed(43)
    model, routes, source = model_and_payload()
    gate = model.layers[0]
    inputs = torch.randn(3, 8)
    mask = torch.ones(3, dtype=torch.bool)
    native_weights, native_indices, _ = gate(inputs, mask, None)
    with automodel_router_replay_context(
        model, routes, source, enabled=True
    ) as assignments:
        weights, indices, _ = gate(inputs, mask, None)
        scores = torch.nn.functional.linear(inputs, gate.weight).float().softmax(-1)
        expected = scores.gather(-1, indices)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected)
        with automodel_router_replay_context(model, None, None, enabled=False):
            reference_weights, reference_indices, _ = gate(inputs, mask, None)
        assert torch.equal(reference_indices, native_indices)
        torch.testing.assert_close(reference_weights, native_weights)
        verify_automodel_router_replay(assignments)
