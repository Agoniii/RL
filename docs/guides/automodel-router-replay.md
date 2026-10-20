# AutoModel routing replay: experimental base implementation

This implementation replays vLLM's discrete expert choices during policy
logprob inference and training. The Gate recomputes routing weights from the
current model parameters. Reference-policy inference uses native routing.

The base path is intentionally limited to **Qwen3 MoE, AutoModel DTensor V2,
synchronous single-turn GRPO, and direct vLLM transport**. Training requires
explicit TP=1, CP=1, EP=1, sequence parallel off, packing off, and dynamic
batching off. FSDP data parallelism remains available but needs GPU acceptance
testing. The driver rejects asynchronous GRPO, TQ, Gym, and multiple rollout
turns. Workers independently validate the model and parallel configuration.

## Required configuration

See
[`grpo-qwen3-30ba3b-1n8g-automodel-r3-base.yaml`](../../examples/configs/recipes/llm/grpo-qwen3-30ba3b-1n8g-automodel-r3-base.yaml)
for the complete experimental smoke recipe. It explicitly disables auxiliary
loss, routing bias updates, prefix caching, fake-balanced routing, and model
compilation and scoped CUDA graphs. AutoModel must use its custom Qwen3 MoE implementation, not the
HuggingFace fallback.

The AutoModel submodule requires the model-scoped `RouterReplay.replay_target`
API and the independent `replay_selection` observer in the dependency patch
paired with this change. The adapter rejects older handles. The submodule pins
the published [AutoModel dependency commit
`9cfde067`](https://github.com/NVIDIA-NeMo/Automodel/commit/9cfde0673181a363cb879373de00804eade5fc62)
on `Agoniii/feat/r3-scoped-replay`, based on AutoModel `72daceff`.
This is an experimental dependency branch, not a released AutoModel feature.

This draft was prepared against RL `612d5274`. Upstream subsequently added
[AutoModel R3 in #4458](https://github.com/NVIDIA-NeMo/RL/pull/4458).
Reconcile this implementation's snapshot, provenance, and replay verification
contracts with that path and the current AutoModel dependency before merge.
The CPU results below do not validate that pending integration.

## Correctness boundary

Each generation request produces an immutable snapshot containing its complete
prefix and response tokens, response boundary, generation logprobs, route
tensor, route-source tags, request ID, and policy version. A digest detects
accidental changes. Before policy inference or training, the snapshot must
match the batch's tokens, length, and response logprobs. Every supervised
position must belong to that response. The base path rejects a second request
instead of combining routes from different generations.

Route source is independent of expert ID. Captured rows must reproduce the
captured target. Missing rows must use native Gate selection from the first
forward. Checkpoint recomputation must reproduce that first effective
selection. The observer receives native selection before replay applies its
override, so substituting arbitrary legal experts for fallback fails.
Verification also fails if an expected Gate was never observed.

The terminal input token and right padding use deterministic legal expert IDs.
Those choices do not represent captured coverage. Auxiliary loss and routing
bias updates must be zero: a language-model loss mask alone cannot prevent
an auxiliary autograd function from injecting gradients. Entirely masked global
batches skip the optimizer and scheduler. Masked samples carry padding
provenance; valid samples retain their original masks and normalization.

Replay contexts remain active through backward and restore handle state on
exceptions. Reference inference temporarily disables replay and restores the
policy weights and sampling parameters even if inference fails.

This enforces selection consistency for captured rows, with explicit native
fallback for missing rows. It does **not** establish equality of vLLM and
AutoModel numerics, convergence, throughput, or fully captured routing when
vLLM reports missing rows.

## Validation

CPU contract tests:

```bash
uv run pytest --confcutdir=tests/unit/models/automodel \
  tests/unit/models/automodel/test_router_replay_base.py \
  tests/unit/models/automodel/test_router_replay_real_gate.py -q -o addopts=
```

The contract suite uses the real AutoModel replay primitive and a differentiable
Gate double; the second file exercises the actual AutoModel Gate. They cover
snapshot integrity, single-turn rejection, configuration
guards, layer mapping, reference isolation, missing-route negative tests, and
padding/dummy gradient invariance with reentrant and non-reentrant checkpointing.
The real-Gate suite also compares selected live weights and verifies reference
isolation. These CPU tests do not run a full AutoModel model or vLLM.

The dependency's real Gate tests live at
`3rdparty/Automodel-workspace/Automodel/tests/unit_tests/moe/test_router_replay.py`.
Run them in the supported AutoModel environment.

The tiny full-model CUDA test checks two batch rows, two MoE layers, causal
padding, nonzero router gradients, and backward with checkpointing on/off:

```bash
uv run pytest --confcutdir=tests/functional \
  tests/functional/test_automodel_router_replay.py -q -o addopts=
```

The pinned Qwen3 MoE constructor requires CUDA even with torch expert kernels;
the test skips on a CPU-only installation. A skip is not validation.

The GPU execution smoke test is:

```bash
bash tests/test_suites/llm/grpo-qwen3-30ba3b-1n8g-automodel-r3-base.sh
```

The recipe defaults to two steps and seed 42. Override the step count with
`R3_SMOKE_STEPS`. `TEST_DRYRUN=1` checks recipe discovery through the existing
test-suite harness. Use a Linux checkout path without spaces, as required by
that harness. This driver checks completed-step count and finite loss/nonzero
gradient metrics. Runtime replay assertions check the selected expert IDs.
It is not enrolled in nightly CI until GPU acceptance has passed.

Before merge, retain evidence from the repository's pinned Linux environment
for all of the following:

1. Real Gate and tiny Qwen3 MoE forward/backward tests, including live router
   gradients, native reference isolation, and AC on/off.
2. vLLM generation → snapshot transport → policy logprobs → backward/update
   over at least two refits. Record capture/fallback coverage and missing
   routes explicitly.
3. Real-model gradient comparison after adding right padding or masked samples,
   including an entirely masked batch that changes neither optimizer state nor
   scheduler state.
4. R3-off regression and an independent all-captured replay comparison.

Multi-turn per-request training views, CP/EP/TP expansion, packing, TQ/Gym,
asynchronous rollout, auxiliary losses, and the full distributed trace checker
are follow-up work. Their absence does not justify relaxing these base checks.
