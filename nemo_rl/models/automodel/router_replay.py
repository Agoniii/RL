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

"""Model-scoped rollout routing replay for the AutoModel backend."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, cast

import torch
from torch import nn

from nemo_rl.experience.r3 import RouteSource
from nemo_rl.models.router_replay import (
    router_replay_enabled,
    validate_automodel_r3_scope,
)

if TYPE_CHECKING:
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.router_replay import RouterReplay

    from nemo_rl.models.policy import PolicyConfig

ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL = -1

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SUPPORTED_MODEL_TYPES = {"qwen3_moe"}


@dataclass(frozen=True)
class RouterReplayLayer:
    """One decoder MoE gate and its rollout-payload mapping."""

    decoder_layer_id: int
    payload_layer_index: int
    gate: Gate
    handle: RouterReplay
    expert_count: int
    top_k: int


@dataclass
class RouterReplayAssignment:
    """A validated per-layer replay target installed for one microbatch."""

    layer: RouterReplayLayer
    target: torch.Tensor
    source: torch.Tensor
    first_native: torch.Tensor | None = None
    first_effective: torch.Tensor | None = None
    observations: int = 0

    def observe(self, native: torch.Tensor, effective: torch.Tensor) -> None:
        """Verify independently observed values passed from replay to Gate."""
        target = self.target.to(effective.device)
        missing = target.eq(-1).all(dim=-1)
        if effective.shape != target.shape or native.shape != target.shape:
            raise RuntimeError("AutoModel Gate replay evidence shape mismatch")
        if self.observations == 0:
            if not torch.equal(effective[~missing], target[~missing]):
                raise RuntimeError("AutoModel captured-to-target replay mismatch")
            if not torch.equal(effective[missing], native[missing]):
                raise RuntimeError("AutoModel missing-to-native fallback mismatch")
            self.first_native = native.detach().clone()
            self.first_effective = effective.detach().clone()
        elif self.first_effective is None or not torch.equal(
            effective, self.first_effective
        ):
            raise RuntimeError("AutoModel recompute-to-first-effective mismatch")
        self.observations += 1


def validate_automodel_router_replay_config(
    config: PolicyConfig, model_config: Any
) -> None:
    """Validate and prepare AutoModel construction arguments for R3."""
    if not router_replay_enabled(config):
        return
    validate_automodel_r3_scope(config)

    generation = config.get("generation") or {}
    dtensor_cfg = config.get("dtensor_cfg") or {}
    megatron_cfg = config.get("megatron_cfg") or {}
    if generation.get("backend") != "vllm":
        raise ValueError("router_replay.enabled requires vLLM generation.")
    if megatron_cfg.get("enabled", False):
        raise ValueError(
            "AutoModel router replay requires policy.megatron_cfg.enabled=false."
        )
    if not dtensor_cfg.get("enabled", False) or not dtensor_cfg.get("_v2", False):
        raise ValueError(
            "AutoModel router replay requires the DTensor V2 policy backend."
        )

    model_type = getattr(model_config, "model_type", None)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "AutoModel router replay currently supports only Qwen3 MoE; "
            f"got model_type={model_type!r}."
        )

    automodel_kwargs = dtensor_cfg.get("automodel_kwargs")
    if not isinstance(automodel_kwargs, dict):
        raise ValueError(
            "AutoModel router replay requires policy.dtensor_cfg.automodel_kwargs."
        )
    if automodel_kwargs.get("force_hf") is True:
        raise ValueError("AutoModel router replay is unavailable with force_hf=true.")
    if automodel_kwargs.get("compile_config") is not None:
        raise ValueError("AutoModel R3 does not yet support compiled model execution")
    backend = automodel_kwargs.get("backend")
    fake_balanced = (
        backend.get("fake_balanced_gate", False)
        if isinstance(backend, dict)
        else getattr(backend, "fake_balanced_gate", False)
    )
    if fake_balanced:
        raise ValueError("AutoModel R3 does not support fake_balanced_gate")
    cuda_graph = (
        backend.get("cuda_graph")
        if isinstance(backend, dict)
        else getattr(backend, "cuda_graph", None)
    )
    graph_modules = (
        cuda_graph.get("modules")
        if isinstance(cuda_graph, dict)
        else getattr(cuda_graph, "modules", None)
    )
    if graph_modules:
        raise ValueError("AutoModel R3 base requires CUDA graph capture disabled")
    if dtensor_cfg.get("tensor_parallel_size", 1) != 1 or dtensor_cfg.get(
        "sequence_parallel", False
    ):
        raise ValueError(
            "AutoModel R3 currently requires TP=1 and sequence_parallel=false"
        )
    vllm_cfg = generation.get("vllm_cfg") or {}
    if vllm_cfg.get("enable_prefix_caching") is not False or (
        generation.get("vllm_kwargs") or {}
    ).get("enable_prefix_caching", False):
        raise ValueError(
            "AutoModel R3 requires vllm_cfg.enable_prefix_caching=false until KV provenance is captured"
        )

    overrides = automodel_kwargs.get("moe_overrides")
    if not isinstance(overrides, dict):
        raise ValueError(
            "AutoModel router replay requires explicit moe_overrides with "
            "aux_loss_coeff=0 and gate_bias_update_factor=0."
        )
    for key in ("aux_loss_coeff", "gate_bias_update_factor"):
        if key not in overrides or float(overrides[key]) != 0.0:
            raise ValueError(
                "AutoModel router replay requires explicit "
                f"moe_overrides.{key}=0; got {overrides.get(key)!r}."
            )
    if overrides.get("enable_routing_replay") is False:
        raise ValueError(
            "router_replay.enabled conflicts with "
            "automodel_kwargs.moe_overrides.enable_routing_replay=false."
        )
    overrides["enable_routing_replay"] = True


def _decoder_layer_id(module_name: str) -> int | None:
    match = _LAYER_PATTERN.search(module_name)
    return int(match.group(1)) if match is not None else None


def _candidate_layers(model: nn.Module) -> list[tuple[int, Gate, RouterReplay]]:
    candidates: list[tuple[int, Gate, RouterReplay]] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        handle = getattr(module, "router_replay", None)
        layer_id = _decoder_layer_id(name)
        if hasattr(module, "router_replay") and layer_id is not None and handle is None:
            raise ValueError(f"Decoder Gate {name} is missing its replay handle")
        if handle is None or layer_id is None or id(handle) in seen:
            continue
        seen.add(id(handle))
        candidates.append(
            (layer_id, cast("Gate", module), cast("RouterReplay", handle))
        )
    candidates.sort(key=lambda item: item[0])
    if len({layer for layer, _, _ in candidates}) != len(candidates):
        raise ValueError("Multiple replay gates map to the same decoder layer")
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "model_type", None) == "qwen3_moe":
        dense_layers = set(getattr(config, "mlp_only_layers", []) or [])
        stride = int(getattr(config, "decoder_sparse_step", 1))
        expected = {
            index
            for index in range(config.num_hidden_layers)
            if index not in dense_layers and (index + 1) % stride == 0
        }
        if {layer for layer, _, _ in candidates} != expected:
            raise ValueError(
                "AutoModel replay Gate coverage differs from the Qwen3 MoE configuration"
            )
    for _, _, handle in candidates:
        if not callable(getattr(handle, "replay_target", None)) or not hasattr(
            handle, "selection_observer"
        ):
            raise ValueError(
                "AutoModel RouterReplay lacks the required model-scoped replay/evidence API"
            )
    return candidates


def automodel_router_replay_dimensions(model: nn.Module) -> tuple[int, int]:
    """Return the compressed rollout payload dimensions ``(layers, top_k)``."""
    candidates = _candidate_layers(model)
    if not candidates:
        raise ValueError(
            "No model-owned AutoModel RouterReplay handles were found. Ensure "
            "moe_overrides.enable_routing_replay=true uses the custom Qwen3 MoE model."
        )
    top_ks = {int(getattr(gate, "topk")) for _, gate, _ in candidates}
    if len(top_ks) != 1:
        raise ValueError(
            f"AutoModel replay gates have inconsistent top-k values: {top_ks}"
        )
    return len(candidates), top_ks.pop()


def validate_automodel_router_replay_model(model: nn.Module) -> None:
    """Verify every replay-enabled Gate obeys the initial R3 loss boundary."""
    candidates = _candidate_layers(model)
    if not candidates:
        automodel_router_replay_dimensions(model)
    for layer_id, gate, _ in candidates:
        aux = float(gate.aux_loss_coeff)
        bias = float(gate.bias_update_factor)
        if aux != 0.0 or bias != 0.0:
            raise ValueError(
                "AutoModel router replay requires auxiliary loss and gate bias "
                f"updates disabled; layer={layer_id}, aux_loss_coeff={aux}, "
                f"bias_update_factor={bias}."
            )


def _normalize_routes(routed_experts: torch.Tensor) -> torch.Tensor:
    """Flatten [B,S,L,K] to [B*S,L,K] in AutoModel's batch-major order."""
    if routed_experts.dim() == 4:
        return routed_experts.reshape(
            routed_experts.shape[0] * routed_experts.shape[1],
            routed_experts.shape[2],
            routed_experts.shape[3],
        )
    raise ValueError(
        "routed_experts must have shape [B, S, L, K]; "
        f"got {tuple(routed_experts.shape)}"
    )


def _validate_target(target: torch.Tensor, layer: RouterReplayLayer) -> None:
    if target.shape[-1] != layer.top_k:
        raise ValueError(
            f"Replay top-k mismatch for layer {layer.decoder_layer_id}: "
            f"payload={target.shape[-1]}, gate={layer.top_k}."
        )
    negative = target.lt(0)
    missing = target.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).all(dim=-1)
    invalid = negative.any(dim=-1) & ~missing
    if bool(invalid.any().item()):
        row = int(invalid.nonzero()[0].item())
        raise ValueError(
            "Replay rows must contain global expert IDs or the all--1 missing "
            f"sentinel; layer={layer.decoder_layer_id}, row={row}, "
            f"value={target[row].detach().cpu().tolist()}."
        )
    captured = target[~missing]
    if captured.numel() == 0:
        return
    if int(captured.max().item()) >= layer.expert_count:
        raise ValueError(
            f"Replay expert ID exceeds layer {layer.decoder_layer_id} expert count "
            f"{layer.expert_count}."
        )
    sorted_target = captured.sort(dim=-1).values
    if bool((sorted_target[:, 1:] == sorted_target[:, :-1]).any().item()):
        raise ValueError(
            f"Replay target contains duplicate top-k experts at layer {layer.decoder_layer_id}."
        )


def build_automodel_router_replay_assignments(
    model: nn.Module,
    routed_experts: torch.Tensor,
    route_source: torch.Tensor,
) -> list[RouterReplayAssignment]:
    """Map canonical route payloads to model-owned AutoModel Gate handles."""
    routes = _normalize_routes(routed_experts)
    if routes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Replay expert IDs must have a signed integer dtype")
    sources = _normalize_routes(route_source.unsqueeze(-1)).squeeze(-1)
    if sources.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Replay source metadata must have a signed integer dtype")
    if tuple(sources.shape) != tuple(routes.shape[:-1]):
        raise ValueError(
            "route_source must align with routed_experts token/layer axes; "
            f"source={tuple(sources.shape)}, routes={tuple(routes.shape)}."
        )
    candidates = _candidate_layers(model)
    if not candidates:
        automodel_router_replay_dimensions(model)

    total_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers"))
    if routes.shape[1] == len(candidates):
        payload_indices = {
            layer_id: idx for idx, (layer_id, _, _) in enumerate(candidates)
        }
    elif routes.shape[1] == total_layers:
        payload_indices = {layer_id: layer_id for layer_id, _, _ in candidates}
    else:
        raise ValueError(
            "routed_experts layer axis does not match compressed MoE or full "
            f"decoder layout: payload={routes.shape[1]}, moe={len(candidates)}, "
            f"decoder={total_layers}."
        )

    assignments: list[RouterReplayAssignment] = []
    for layer_id, gate, handle in candidates:
        layer = RouterReplayLayer(
            decoder_layer_id=layer_id,
            payload_layer_index=payload_indices[layer_id],
            gate=gate,
            handle=handle,
            expert_count=int(getattr(gate, "n_experts")),
            top_k=int(getattr(gate, "topk")),
        )
        target = routes[:, layer.payload_layer_index, :].to(
            device=next(gate.parameters()).device,
            dtype=torch.long,
        )
        source = sources[:, layer.payload_layer_index].to(target.device)
        # Padding can arrive as zeros from generic transport/packing. It has
        # no capture semantics and must never supply duplicate top-k IDs.
        unused = source.eq(int(RouteSource.PADDING)) | source.eq(
            int(RouteSource.TERMINAL_UNUSED)
        )
        target = target.clone()
        target[unused] = torch.arange(layer.top_k, device=target.device)
        _validate_target(target, layer)
        missing_target = target.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).all(dim=-1)
        captured_source = source.eq(int(RouteSource.CAPTURED))
        fallback_source = source.eq(int(RouteSource.MISSING)) | source.eq(
            int(RouteSource.UNCAPTURED_CONTEXT)
        )
        unused_source = source.eq(int(RouteSource.TERMINAL_UNUSED)) | source.eq(
            int(RouteSource.PADDING)
        )
        known_source = captured_source | fallback_source | unused_source
        if not bool(known_source.all().item()):
            raise ValueError(
                f"Unknown route_source value at layer {layer.decoder_layer_id}."
            )
        if bool((captured_source & missing_target).any().item()):
            raise ValueError(
                f"CAPTURED route_source has no captured target at layer {layer.decoder_layer_id}."
            )
        if bool((fallback_source & ~missing_target).any().item()):
            raise ValueError(
                "Missing/uncaptured route_source must install the all--1 native "
                f"fallback target at layer {layer.decoder_layer_id}."
            )
        assignments.append(
            RouterReplayAssignment(layer=layer, target=target, source=source)
        )
    return assignments


def verify_automodel_router_replay(
    assignments: list[RouterReplayAssignment],
) -> None:
    """Check captured, native-fallback, and recompute replay evidence."""
    for assignment in assignments:
        native = assignment.first_native
        effective = assignment.first_effective
        apply_count = assignment.observations
        if native is None or effective is None or apply_count < 1:
            raise RuntimeError(
                "AutoModel Gate did not expose independent native/effective R3 "
                f"evidence for layer {assignment.layer.decoder_layer_id}."
            )
        target = assignment.target.to(effective.device)
        missing = target.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).all(dim=-1)
        if not torch.equal(effective[~missing], target[~missing]):
            raise RuntimeError(
                f"AutoModel captured-to-target replay mismatch at layer {assignment.layer.decoder_layer_id}."
            )
        if not torch.equal(effective[missing], native.to(effective.device)[missing]):
            raise RuntimeError(
                f"AutoModel missing-to-native fallback mismatch at layer {assignment.layer.decoder_layer_id}."
            )


@contextmanager
def automodel_router_replay_context(
    model: nn.Module,
    routed_experts: torch.Tensor | None,
    route_source: torch.Tensor | None,
    *,
    enabled: bool,
) -> Iterator[list[RouterReplayAssignment]]:
    """Install model-scoped replay targets for one forward/backward lifetime."""
    if not enabled:
        handles = [
            cast("RouterReplay", module.router_replay)
            for module in model.modules()
            if getattr(module, "router_replay", None) is not None
        ]
        states = [(handle.mode, handle.selection_observer) for handle in handles]
        try:
            for handle in handles:
                handle.mode = None
                handle.selection_observer = None
            yield []
        finally:
            for handle, (mode, observer) in zip(handles, states):
                handle.mode, handle.selection_observer = mode, observer
        return
    if routed_experts is None:
        raise ValueError(
            "AutoModel router replay requires routed_experts in the batch."
        )
    if route_source is None:
        raise ValueError(
            "AutoModel router replay requires explicit route_source metadata."
        )

    assignments = build_automodel_router_replay_assignments(
        model, routed_experts, route_source
    )
    with ExitStack() as stack:
        for assignment in assignments:
            stack.enter_context(
                assignment.layer.handle.replay_target(
                    assignment.target, selection_observer=assignment.observe
                )
            )
        yield assignments
