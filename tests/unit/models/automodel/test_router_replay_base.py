# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU contract tests; the Gate double does not validate distributed kernels."""

import json

from types import SimpleNamespace

import pytest

import torch

from torch import nn

from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.moe.router_replay import RouterReplay, replay_selection

from nemo_rl.experience.r3 import (
    RouteSource,
    build_route_source,
    decode_route_snapshot,
    encode_route_snapshot,
    prepare_single_turn_replay,
)
from nemo_rl.models.router_replay import validate_automodel_r3_driver


from nemo_rl.models.automodel.router_replay import (
    automodel_router_replay_context,
    build_automodel_router_replay_assignments,
    validate_automodel_router_replay_config,
    validate_automodel_router_replay_model,
    verify_automodel_router_replay,
)


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
        self.experts = nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
        self.router_replay = RouterReplay()
        self.n_experts, self.topk = 4, 2
        self.aux_loss_coeff = self.bias_update_factor = 0.0

    def forward(self, inputs):
        scores = (inputs @ self.weight.T).softmax(-1)
        selected = replay_selection(self.router_replay, scores.topk(2).indices)
        return (
            scores.gather(-1, selected)
            * (inputs.unsqueeze(1) * self.experts[selected]).sum(-1)
        ).sum(-1)


class Model(nn.Module):
    def __init__(self, layers=1):
        super().__init__()
        self.layers = nn.ModuleList([Gate() for _ in range(layers)])
        self.config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=layers)


def payload(rows=3, layers=1):
    routes = torch.tensor([0, 1], dtype=torch.int16).expand(rows, layers, 2).clone()
    source = torch.full((rows, layers), int(RouteSource.CAPTURED), dtype=torch.int8)
    routes[1] = -1
    source[1] = int(RouteSource.MISSING)
    return routes.unsqueeze(0), source.unsqueeze(0)


def snapshot(tokens, start, expert, request):
    routes = (
        torch.tensor([expert, (expert + 1) % 4], dtype=torch.int16)
        .expand(len(tokens), 1, 2)
        .clone()
    )
    source = torch.full((len(tokens), 1), int(RouteSource.CAPTURED), dtype=torch.int8)
    source[-1] = int(RouteSource.TERMINAL_UNUSED)
    return encode_route_snapshot(
        request_id=request,
        policy_version=7,
        token_ids=torch.tensor(tokens),
        response_start=start,
        generation_logprobs=torch.full((len(tokens) - start,), -0.5),
        routed_experts=routes,
        route_source=source,
    )


@pytest.mark.parametrize("bad", [[0, -1], [-2, -2], [0, 4], [1, 1]])
def test_invalid_expert_rows_fail(bad):
    model = Model()
    routes, source = payload()
    routes[0, 0, 0] = torch.tensor(bad)
    with pytest.raises(ValueError):
        build_automodel_router_replay_assignments(model, routes, source)


def test_legal_dummy_ids_cannot_masquerade_as_missing_capture():
    model = Model()
    routes, source = payload()
    source[0, 0] = int(RouteSource.UNCAPTURED_CONTEXT)
    with pytest.raises(ValueError, match="native fallback"):
        build_automodel_router_replay_assignments(model, routes, source)


def test_model_scoped_replay_reference_isolation_and_exception_cleanup():
    model, other = Model(), Model()
    routes, source = payload()
    native = torch.tensor([[2, 3], [1, 3], [3, 2]])
    handle = model.layers[0].router_replay
    with pytest.raises(RuntimeError, match="injected"):
        with automodel_router_replay_context(
            model, routes, source, enabled=True
        ) as assignments:
            effective = replay_selection(handle, native)
            assert torch.equal(effective[1], native[1])
            assert other.layers[0].router_replay.mode is None
            with automodel_router_replay_context(model, None, None, enabled=False):
                assert torch.equal(
                    replay_selection(handle, native.flip(0)), native.flip(0)
                )
            assert assignments[0].observations == 1
            verify_automodel_router_replay(assignments)
            raise RuntimeError("injected")
    assert handle.mode is None and handle.selection_observer is None
    assert handle.effective_indices is None


def test_independent_observer_rejects_legal_but_wrong_fallback(monkeypatch):
    model = Model()
    routes, source = payload()
    handle = model.layers[0].router_replay
    monkeypatch.setattr(
        handle, "apply", lambda native: torch.tensor([[0, 1], [0, 1], [0, 1]])
    )
    with pytest.raises(RuntimeError, match="missing-to-native"):
        with automodel_router_replay_context(model, routes, source, enabled=True):
            replay_selection(handle, torch.tensor([[2, 3], [2, 3], [2, 3]]))


def test_recompute_reuses_first_fallback_even_if_native_changes():
    model = Model()
    routes, source = payload()
    handle = model.layers[0].router_replay
    native = torch.tensor([[2, 3], [1, 3], [3, 2]])
    with automodel_router_replay_context(
        model, routes, source, enabled=True
    ) as assignments:
        first = replay_selection(handle, native).clone()
        assert torch.equal(replay_selection(handle, native.flip(0).flip(1)), first)
        assert assignments[0].observations == 2
        verify_automodel_router_replay(assignments)


def test_padding_dummy_and_checkpoint_do_not_change_valid_gradients():
    torch.manual_seed(12)
    model = Model()
    inputs = torch.randn(3, 3, dtype=torch.float64)
    routes, source = payload()
    gate = model.layers[0]

    def gradients(padding, ac, dummy, reentrant=True):
        model.zero_grad(set_to_none=True)
        x = torch.cat(
            [inputs, torch.randn(padding, 3, dtype=torch.float64)]
        ).requires_grad_()
        r = torch.cat(
            [routes.squeeze(0), torch.full((padding, 1, 2), -1, dtype=torch.int16)]
        )
        s = torch.cat(
            [
                source.squeeze(0),
                torch.full((padding, 1), int(RouteSource.PADDING), dtype=torch.int8),
            ]
        )
        with automodel_router_replay_context(
            model, r.unsqueeze(0), s.unsqueeze(0), enabled=True
        ) as assignments:
            output = checkpoint(gate, x, use_reentrant=reentrant) if ac else gate(x)
            output[:3].square().sum().backward()
            verify_automodel_router_replay(assignments)
            if ac:
                assert assignments[0].observations == 2
        if dummy:
            with automodel_router_replay_context(
                model, r.unsqueeze(0), s.unsqueeze(0), enabled=True
            ):
                (gate(x).sum() * 0).backward()
        return [p.grad.clone() for p in model.parameters()]

    reference = gradients(0, False, False)
    for actual in (
        gradients(4, False, True),
        gradients(4, True, True),
        gradients(4, True, True, False),
    ):
        for expected, value in zip(reference, actual):
            torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
    assert reference[0].abs().sum() > 0  # Router weights still learn.


def test_model_auxiliary_and_missing_gate_guards():
    model = Model(2)
    model.layers[0].aux_loss_coeff = 0.01
    with pytest.raises(ValueError, match="auxiliary"):
        validate_automodel_router_replay_model(model)
    model.layers[1] = nn.Identity()
    with pytest.raises(ValueError, match="coverage"):
        validate_automodel_router_replay_model(model)


def test_compressed_and_full_layer_axes_map_identically():
    model = Model(4)
    model.layers[0] = model.layers[2] = nn.Identity()
    model.config.decoder_sparse_step = 2
    routes, source = payload(rows=6, layers=2)
    routes[:, :, 1] = torch.tensor([2, 3])
    source[:, :, 1] = int(RouteSource.CAPTURED)
    compact = build_automodel_router_replay_assignments(
        model, routes.reshape(2, 3, 2, 2), source.reshape(2, 3, 2)
    )
    full_routes = torch.zeros(6, 4, 2, dtype=torch.int16)
    full_source = torch.full((6, 4), int(RouteSource.PADDING), dtype=torch.int8)
    full_routes[:, [1, 3]] = routes.squeeze(0)
    full_source[:, [1, 3]] = source.squeeze(0)
    full = build_automodel_router_replay_assignments(
        model, full_routes.reshape(2, 3, 4, 2), full_source.reshape(2, 3, 4)
    )
    assert [a.layer.decoder_layer_id for a in full] == [1, 3]
    for a, b in zip(compact, full):
        assert torch.equal(a.target, b.target)


def config():
    return {
        "router_replay": {"enabled": True},
        "megatron_cfg": {"enabled": False},
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {"enable_prefix_caching": False, "async_engine": False},
        },
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_parallel_size": 1,
            "sequence_parallel": False,
            "automodel_kwargs": {
                "moe_overrides": {"aux_loss_coeff": 0, "gate_bias_update_factor": 0}
            },
        },
    }


def data():
    encoded = snapshot([1, 2, 3, 4], 2, 0, "first")
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0, 0]]),
        "input_lengths": torch.tensor([4]),
        "token_mask": torch.tensor([[0, 0, 1, 1, 0, 0]]),
        "sample_mask": torch.ones(1),
        "generation_logprobs": torch.full((1, 6), -0.5),
        "r3_snapshots": [json.dumps([encoded])],
    }


def test_single_turn_snapshot_roundtrip_padding_and_masked_sample():
    batch = data()
    routes, source = prepare_single_turn_replay(batch)
    assert routes.shape == (1, 6, 1, 2)
    assert routes[0, 0, 0].tolist() == [0, 1]
    assert source[0, 3, 0] == RouteSource.TERMINAL_UNUSED
    assert source[0, 4:].eq(RouteSource.PADDING).all()
    routes.zero_()
    assert prepare_single_turn_replay(batch)[0][0, 0, 0].tolist() == [0, 1]
    batch["sample_mask"].zero_()
    assert prepare_single_turn_replay(batch)[1].eq(RouteSource.PADDING).all()


def test_different_prefix_routes_from_two_requests_are_rejected():
    batch = data()
    first = json.loads(batch["r3_snapshots"][0])[0]
    second = snapshot([1, 2, 3, 4], 2, 2, "second")
    assert not torch.equal(
        decode_route_snapshot(first).routed_experts,
        decode_route_snapshot(second).routed_experts,
    )
    batch["r3_snapshots"] = [json.dumps([first, second])]
    with pytest.raises(ValueError, match="exactly one request"):
        prepare_single_turn_replay(batch)


@pytest.mark.parametrize("field", ["input_ids", "generation_logprobs", "token_mask"])
def test_mismatched_training_view_fails(field):
    batch = data()
    position = 2 if field == "generation_logprobs" else 0
    batch[field][0, position] += 1
    with pytest.raises(ValueError):
        prepare_single_turn_replay(batch)


def test_tampered_snapshot_fails():
    batch = data()
    encoded = json.loads(json.loads(batch["r3_snapshots"][0])[0])
    encoded["response_start"] = 1
    with pytest.raises(ValueError, match="digest"):
        decode_route_snapshot(json.dumps(encoded))


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("expert_parallel_size", 2),
        ("sequence_parallel", True),
        ("_v2", False),
        ("expert_parallel_size", None),
    ],
)
def test_parallel_scope_fails_early(field, value):
    cfg = config()
    cfg["dtensor_cfg"][field] = value
    with pytest.raises(ValueError):
        validate_automodel_router_replay_config(
            cfg, SimpleNamespace(model_type="qwen3_moe")
        )


@pytest.mark.parametrize(
    "field", ["async_training", "use_tq", "use_gym", "max_rollout_turns"]
)
def test_driver_scope_fails_early(field):
    args = dict(async_training=False, use_tq=False, use_gym=False, max_rollout_turns=1)
    args[field] = 2 if field == "max_rollout_turns" else True
    with pytest.raises(ValueError):
        validate_automodel_r3_driver(config(), **args)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("sequence_packing", "enabled", True),
        ("dynamic_batching", "enabled", True),
        ("vllm_cfg", "async_engine", True),
        ("vllm_cfg", "enable_prefix_caching", True),
        ("moe_overrides", "aux_loss_coeff", 0.01),
        ("moe_overrides", "gate_bias_update_factor", 0.01),
        ("moe_overrides", "enable_routing_replay", False),
        ("automodel_kwargs", "force_hf", True),
        ("automodel_kwargs", "compile_config", {}),
        ("automodel_kwargs", "backend", {"fake_balanced_gate": True}),
        ("automodel_kwargs", "backend", {"cuda_graph": {"modules": ["moe_router"]}}),
    ],
)
def test_unsupported_configuration_fails(section, key, value):
    cfg = config()
    sections = {
        **cfg,
        "vllm_cfg": cfg["generation"]["vllm_cfg"],
        "automodel_kwargs": cfg["dtensor_cfg"]["automodel_kwargs"],
        "moe_overrides": cfg["dtensor_cfg"]["automodel_kwargs"]["moe_overrides"],
    }
    sections[section][key] = value
    with pytest.raises(ValueError):
        validate_automodel_router_replay_config(
            cfg, SimpleNamespace(model_type="qwen3_moe")
        )


def test_aux_disable_is_explicit_and_replay_off_does_not_require_new_fields():
    cfg = config()
    validate_automodel_router_replay_config(
        cfg, SimpleNamespace(model_type="qwen3_moe")
    )
    assert cfg["dtensor_cfg"]["automodel_kwargs"]["moe_overrides"][
        "enable_routing_replay"
    ]
    del cfg["dtensor_cfg"]["automodel_kwargs"]["moe_overrides"]["aux_loss_coeff"]
    with pytest.raises(ValueError, match="aux_loss_coeff"):
        validate_automodel_router_replay_config(
            cfg, SimpleNamespace(model_type="qwen3_moe")
        )
    validate_automodel_router_replay_config({}, SimpleNamespace(model_type="anything"))


def test_missing_gate_execution_cannot_pass_verification():
    model = Model()
    routes, source = payload()
    with automodel_router_replay_context(
        model, routes, source, enabled=True
    ) as assignments:
        with pytest.raises(RuntimeError, match="did not expose"):
            verify_automodel_router_replay(assignments)


def test_capture_coverage_comes_from_counts_not_legal_expert_values():
    routes = torch.tensor([0, 1]).expand(6, 1, 2)
    source = build_route_source(
        routes, {"expected_routes": 3, "actual_routes": 1}, valid_length=4
    )
    assert source[:, 0].tolist() == [
        RouteSource.CAPTURED,
        RouteSource.MISSING,
        RouteSource.MISSING,
        RouteSource.TERMINAL_UNUSED,
        RouteSource.PADDING,
        RouteSource.PADDING,
    ]
    with pytest.raises(ValueError, match="counts"):
        build_route_source(
            routes, {"expected_routes": 4, "actual_routes": 1}, valid_length=4
        )
