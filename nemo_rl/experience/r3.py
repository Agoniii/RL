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

"""Immutable per-request routing evidence for single-turn AutoModel R3.

Snapshots are serialized strings so later message-log mutations, Ray transport,
and replay-buffer copies cannot alias their tensors. Digests bind the complete
prefix, response logprobs, route provenance, and response boundaries together.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum

import torch

from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
)

SNAPSHOT_VERSION = 1
SNAPSHOT_FIELD = "r3_snapshots"


class RouteSource(IntEnum):
    """Evidence source for one token/layer route, independent of expert IDs."""

    CAPTURED = 1
    MISSING = 2
    UNCAPTURED_CONTEXT = 3
    TERMINAL_UNUSED = 4
    PADDING = 5


def build_route_source(
    routed_experts: torch.Tensor,
    stats: dict[str, int],
    *,
    valid_length: int,
) -> torch.Tensor:
    """Tag aligned [S,L,K] routes using vLLM capture counts, not expert values.

    Returns [S,L] int8 provenance, including terminal and padding exclusions.
    This applies to the contiguous-prefix capture returned by the synchronous
    vLLM alignment helper with prefix caching disabled.
    """
    if not 0 < valid_length <= routed_experts.shape[0]:
        raise ValueError("Invalid vLLM route provenance length")
    if stats["expected_routes"] != valid_length - 1 or stats["actual_routes"] < 0:
        raise ValueError("vLLM capture counts do not match the required route domain")
    source = torch.full(
        routed_experts.shape[:-1],
        int(RouteSource.PADDING),
        dtype=torch.int8,
        device=routed_experts.device,
    )
    expected = stats["expected_routes"]
    captured = min(stats["actual_routes"], expected)
    source[:captured] = int(RouteSource.CAPTURED)
    source[captured:expected] = int(RouteSource.MISSING)
    source[valid_length - 1] = int(RouteSource.TERMINAL_UNUSED)
    return source


@dataclass(frozen=True)
class RouteSnapshot:
    """Decoded request view; tokens/routes use the input-token coordinate."""

    snapshot_id: str
    request_id: str
    policy_version: int
    response_start: int
    token_ids: torch.Tensor
    generation_logprobs: torch.Tensor
    routed_experts: torch.Tensor
    route_source: torch.Tensor


def _encode_vector(tensor: torch.Tensor, dtype: torch.dtype) -> str:
    return base64.b64encode(
        tensor.detach().to(device="cpu", dtype=dtype).contiguous().numpy().tobytes()
    ).decode("ascii")


def _decode_vector(value: str, dtype: torch.dtype) -> torch.Tensor:
    raw = bytearray(base64.b64decode(value, validate=True))
    return (
        torch.frombuffer(raw, dtype=dtype).clone()
        if raw
        else torch.empty(0, dtype=dtype)
    )


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_route_domain(source: torch.Tensor, valid_length: int) -> None:
    """Reject required prefix positions disguised as padding or terminal rows."""
    if source.ndim != 2 or source.shape[0] != valid_length or valid_length < 2:
        raise ValueError(
            "A route snapshot requires [tokens, layers] provenance and at least two tokens"
        )
    required = source[:-1]
    allowed = (
        required.eq(int(RouteSource.CAPTURED))
        | required.eq(int(RouteSource.MISSING))
        | required.eq(int(RouteSource.UNCAPTURED_CONTEXT))
    )
    if not bool(allowed.all()):
        raise ValueError(
            "Required route domain contains unknown, terminal, or padding provenance"
        )
    if not bool(source[-1].eq(int(RouteSource.TERMINAL_UNUSED)).all()):
        raise ValueError(
            "Only the final snapshot input position must be TERMINAL_UNUSED"
        )


def encode_route_snapshot(
    *,
    request_id: str,
    policy_version: int,
    token_ids: torch.Tensor,
    response_start: int,
    generation_logprobs: torch.Tensor,
    routed_experts: torch.Tensor,
    route_source: torch.Tensor,
) -> str:
    """Seal full prefix+response routes with the logprobs from that request."""
    length = token_ids.numel()
    if not request_id or policy_version < 0 or token_ids.ndim != 1:
        raise ValueError(
            "Snapshot requires request identity, nonnegative weight version and 1D tokens"
        )
    if not 0 < response_start < length or generation_logprobs.shape != (
        length - response_start,
    ):
        raise ValueError("Snapshot response span and generation logprobs do not align")
    if routed_experts.ndim != 3 or tuple(route_source.shape) != tuple(
        routed_experts.shape[:-1]
    ):
        raise ValueError("Snapshot routes and provenance do not align")
    validate_route_domain(route_source, length)
    payload = {
        "schema_version": SNAPSHOT_VERSION,
        "request_id": request_id,
        "policy_version": policy_version,
        "response_start": response_start,
        "token_ids": _encode_vector(token_ids, torch.int64),
        "generation_logprobs": _encode_vector(generation_logprobs, torch.float32),
        "routed_experts": encode_routed_experts(routed_experts),
        "route_source": encode_routed_experts(
            route_source.to(torch.int8).unsqueeze(-1)
        ),
    }
    payload["snapshot_id"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return _canonical(payload)


def decode_route_snapshot(encoded: str) -> RouteSnapshot:
    """Check a snapshot's commitment before exposing its tensors."""
    payload = json.loads(encoded)
    expected_keys = {
        "schema_version",
        "request_id",
        "policy_version",
        "response_start",
        "token_ids",
        "generation_logprobs",
        "routed_experts",
        "route_source",
        "snapshot_id",
    }
    if not isinstance(payload, dict) or payload.keys() != expected_keys:
        raise ValueError("Invalid R3 snapshot fields")
    digest = payload.pop("snapshot_id")
    if (
        payload["schema_version"] != SNAPSHOT_VERSION
        or hashlib.sha256(_canonical(payload).encode()).hexdigest() != digest
    ):
        raise ValueError("R3 snapshot version or digest mismatch")
    tokens = _decode_vector(payload["token_ids"], torch.int64)
    logprobs = _decode_vector(payload["generation_logprobs"], torch.float32)
    routes = decode_routed_experts(payload["routed_experts"], torch.int16)
    source = decode_routed_experts(payload["route_source"], torch.int8).squeeze(-1)
    start = payload["response_start"]
    if (
        not isinstance(payload["request_id"], str)
        or not payload["request_id"]
        or type(payload["policy_version"]) is not int
        or payload["policy_version"] < 0
    ):
        raise ValueError("Invalid R3 snapshot request identity or policy version")
    if (
        type(start) is not int
        or not 0 < start < tokens.numel()
        or logprobs.shape != (tokens.numel() - start,)
    ):
        raise ValueError("Invalid R3 snapshot response span")
    if routes.ndim != 3 or routes.shape[:-1] != source.shape:
        raise ValueError("Invalid R3 snapshot route shape")
    validate_route_domain(source, tokens.numel())
    return RouteSnapshot(
        digest,
        payload["request_id"],
        payload["policy_version"],
        start,
        tokens,
        logprobs,
        routes,
        source,
    )


def prepare_single_turn_replay(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate single-request evidence and build padded policy replay inputs.

    Args:
        data: Batch containing input_ids [B,S], input_lengths [B], token_mask
            [B,S], generation_logprobs [B,S], sample_mask [B], and r3_snapshots
            (B strings, each encoding a JSON list with exactly one snapshot).
            Supervised positions must belong to that request's response.

    Returns:
        Independent tensors on input_ids.device: global expert IDs [B,S,L,K]
        (int16, all -1 for missing capture/padding), and provenance [B,S,L]
        (int8). Masked samples and right padding have PADDING provenance.
        No loss mask, sample weight, input or logprob is modified.
    """
    if SNAPSHOT_FIELD not in data:
        raise ValueError("AutoModel R3 requires an immutable single-request snapshot")
    encoded_rows = data[SNAPSHOT_FIELD]
    ids = data["input_ids"]
    if len(encoded_rows) != ids.shape[0] or ids.shape[0] == 0:
        raise ValueError("Snapshot count differs from the nonempty input batch")
    snapshots: list[RouteSnapshot] = []
    for row, encoded in enumerate(encoded_rows):
        requests = json.loads(encoded)
        if not isinstance(requests, list) or len(requests) != 1:
            raise ValueError(
                "AutoModel R3 base requires exactly one request per trajectory"
            )
        snapshot = decode_route_snapshot(requests[0])
        end = snapshot.token_ids.numel()
        if end != int(data["input_lengths"][row]) or end > ids.shape[1]:
            raise ValueError("Snapshot length differs from the single-turn trajectory")
        if not torch.equal(snapshot.token_ids, ids[row, :end].detach().cpu().long()):
            raise ValueError("Snapshot token IDs differ from the rollout")
        start = snapshot.response_start
        if not torch.equal(
            snapshot.generation_logprobs,
            data["generation_logprobs"][row, start:end].detach().cpu().float(),
        ):
            raise ValueError(
                "Snapshot and generation logprobs belong to different requests"
            )
        mask = data["token_mask"][row].detach().cpu().bool()
        if mask[:start].any() or mask[end:].any():
            raise ValueError("Supervised tokens fall outside the captured response")
        snapshots.append(snapshot)
    layers, top_k = snapshots[0].routed_experts.shape[1:]
    routes = torch.full(
        (*ids.shape, layers, top_k), -1, dtype=torch.int16, device=ids.device
    )
    source = torch.full(
        (*ids.shape, layers),
        int(RouteSource.PADDING),
        dtype=torch.int8,
        device=ids.device,
    )
    for row, snapshot in enumerate(snapshots):
        if snapshot.routed_experts.shape[1:] != (layers, top_k):
            raise ValueError("Snapshot layer/top-k dimensions differ within the batch")
        if not bool(data["sample_mask"][row]):
            continue
        end = snapshot.token_ids.numel()
        routes[row, :end] = snapshot.routed_experts.to(ids.device)
        source[row, :end] = snapshot.route_source.to(ids.device)
    return routes, source
