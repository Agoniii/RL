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

"""Backend-independent router-replay configuration helpers."""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.config import VllmConfig
    from nemo_rl.models.policy import DTensorConfig, PolicyConfig


def router_replay_enabled(config: "PolicyConfig") -> bool:
    """Return whether rollout router replay is enabled for the policy."""
    return bool((config.get("router_replay") or {}).get("enabled", False))


def automodel_router_replay_enabled(config: "PolicyConfig") -> bool:
    """Identify the AutoModel V2 owner without importing either backend."""
    return (
        router_replay_enabled(config)
        and bool((config.get("dtensor_cfg") or {}).get("_v2", False))
        and not bool((config.get("megatron_cfg") or {}).get("enabled", False))
    )


def validate_automodel_r3_scope(config: "PolicyConfig") -> None:
    """Reject combinations outside the synchronous, unpacked V2 base path.

    Required config values come from the exemplar YAML. This check runs both
    on the driver and on the policy worker so direct construction is guarded.
    """
    if not router_replay_enabled(config) or config["megatron_cfg"]["enabled"]:
        return
    dtensor = config["dtensor_cfg"]
    if not dtensor["enabled"]:
        raise ValueError("AutoModel R3 base requires DTensor V2")
    dtensor = cast("DTensorConfig", dtensor)
    if not dtensor["_v2"]:
        raise ValueError("AutoModel R3 base requires DTensor V2")
    for axis in (
        "tensor_parallel_size",
        "context_parallel_size",
        "expert_parallel_size",
    ):
        if dtensor.get(axis) != 1:
            raise ValueError(f"AutoModel R3 base requires explicit {axis}=1")
    if dtensor["sequence_parallel"]:
        raise ValueError("AutoModel R3 base requires sequence_parallel=false")
    if config["sequence_packing"]["enabled"] or config["dynamic_batching"]["enabled"]:
        raise ValueError(
            "AutoModel R3 base requires packing and dynamic batching disabled"
        )
    generation = cast("VllmConfig", config["generation"])
    if generation["backend"] != "vllm" or generation["vllm_cfg"]["async_engine"]:
        raise ValueError("AutoModel R3 base requires synchronous vLLM generation")
    draft = config.get("draft")
    draft_enabled = (
        draft.get("enabled")
        if isinstance(draft, dict)
        else (draft.enabled if draft is not None else False)
    )
    if draft_enabled:
        raise ValueError("AutoModel R3 base does not support draft-model training")


def validate_automodel_r3_driver(
    config: "PolicyConfig",
    *,
    async_training: bool,
    use_tq: bool,
    use_gym: bool,
    max_rollout_turns: int,
) -> None:
    """Validate algorithm/transport scope before allocating worker resources."""
    validate_automodel_r3_scope(config)
    if not automodel_router_replay_enabled(config):
        return
    if async_training or use_tq or use_gym or max_rollout_turns != 1:
        raise ValueError(
            "AutoModel R3 base requires synchronous single-turn GRPO, "
            "data_plane.enabled=false and Gym disabled"
        )


def configure_vllm_for_router_replay(config: "PolicyConfig") -> None:
    """Request routed-expert payloads from vLLM when replay is enabled."""
    if not router_replay_enabled(config):
        return

    generation = cast("VllmConfig", config["generation"])
    vllm_kwargs = generation.setdefault("vllm_kwargs", {})
    vllm_kwargs["enable_return_routed_experts"] = True
    if automodel_router_replay_enabled(config):
        generation["_r3_snapshot_capture"] = True
