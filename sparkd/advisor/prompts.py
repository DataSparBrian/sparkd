from __future__ import annotations

import json
import re

from sparkd.advisor.tool_calls import render_tool_call_block
from sparkd.schemas.advisor import ModDraft, RecipeDraft
from sparkd.schemas.box import BoxCapabilities
from sparkd.schemas.hf import HFModelInfo
from sparkd.schemas.recipe import RecipeSpec


SYSTEM_PROMPT = """You are a vLLM deployment advisor for NVIDIA DGX Spark hardware.

Your job is to translate a Hugging Face model and a target box's hardware capabilities
into a concrete vLLM `serve` recipe (CLI args + env), or to optimize an existing recipe,
or to propose a model-specific patch ("mod") when a model needs a fix to run on vLLM.

Always emit your final answer as a single fenced ```json``` block matching the requested
schema. The recipe `args` keys must be the literal vLLM CLI flag names (e.g.
"--tensor-parallel-size", "--gpu-memory-utilization", "--max-model-len", "--quantization").
Values are strings.

Flag pair rules vLLM enforces — set both together or neither:
- `--tool-call-parser` requires `--enable-auto-tool-choice: "true"`. If the model
  has tool-calling support and you want to expose it, include both. Otherwise
  omit both — never set the parser without enabling auto choice.

Boolean flags (`--enable-auto-tool-choice`, `--trust-remote-code`,
`--enforce-eager`, `--enable-prefix-caching`, etc.) take the string value
"true" in `args`. Sparkd renders them as bare flags on the command line —
do not include `false` for these (omit instead).

Be conservative. Prefer settings that fit comfortably in available VRAM with a margin.
Explain trade-offs in `rationale` in one short paragraph.
"""


def _caps_block(caps: BoxCapabilities) -> str:
    return (
        f"Box capabilities:\n"
        f"- GPU model: {caps.gpu_model}\n"
        f"- GPU count: {caps.gpu_count}\n"
        f"- VRAM per GPU: {caps.vram_per_gpu_gb} GB\n"
        f"- CUDA: {caps.cuda_version or 'unknown'}\n"
        f"- IB iface: {caps.ib_interface or 'none'}\n"
    )


def _model_block(info: HFModelInfo) -> str:
    return (
        f"Hugging Face model facts:\n"
        f"- ID: {info.id}\n"
        f"- Architecture: {info.architecture or 'unknown'}\n"
        f"- Parameters: {info.parameters_b} B\n"
        f"- Context length: {info.context_length}\n"
        f"- Supported dtypes: {', '.join(info.supported_dtypes) or 'unknown'}\n"
        f"- License: {info.license or 'unknown'}\n"
        f"- {render_tool_call_block(info.id)}\n"
    )


def _cluster_block(cluster: dict) -> str:
    nodes = cluster.get("nodes") or []
    n_nodes = len(nodes)
    total_gpus = cluster.get("total_gpus", 0)
    # Per-node GPU count, assuming homogeneous (the common case on a Spark
    # fleet). If heterogeneous, the model can read the per-node breakdown
    # below and decide; for our DGX Spark target every node has 1 GPU.
    gpus_per_node = (
        nodes[0].get("gpu_count", 0) if nodes and n_nodes > 0 else 0
    )

    lines = [
        "Multi-node cluster topology:",
        f"- Cluster: {cluster.get('name', 'unknown')}",
        f"- Nodes: {n_nodes}",
    ]
    for n in nodes:
        lines.append(
            f"  · {n.get('name', '?')}: {n.get('gpu_count', 0)}× "
            f"{n.get('gpu_model') or 'unknown'}, "
            f"{n.get('vram_gb', 0)} GB VRAM, "
            f"IB={n.get('ib') or 'none'}"
        )
    lines.append(f"- Total GPUs across cluster: {total_gpus}")
    lines.append(f"- Aggregate VRAM: {cluster.get('total_vram_gb', 0)} GB")
    lines.append("")
    lines.append(
        f"PARALLELISM REQUIREMENT (binding): "
        f"--tensor-parallel-size × --pipeline-parallel-size MUST equal "
        f"the cluster's total GPU count, which is **{total_gpus}** "
        f"(= {n_nodes} nodes × {gpus_per_node} GPU/node). "
        "Anything less than total_gpus leaves GPUs idle and Ray will "
        "either reject the placement or refuse to schedule workers; "
        "anything more is unsatisfiable."
    )
    lines.append("")
    lines.append(
        f"PREFERRED LAYOUT: tp = {total_gpus}, pp = 1 — distribute "
        f"the model's tensor shards across all {total_gpus} GPUs in one "
        "pipeline stage. The Spark fleet has IB/RoCE between nodes which "
        "supports cross-node tensor-parallel comfortably. Only fall back "
        f"to pp = {n_nodes} (and tp = {gpus_per_node}) when a model's "
        "intermediate-state size genuinely requires the extra "
        "memory-per-stage budget."
    )
    lines.append("")
    lines.append(
        "Set --distributed-executor-backend=ray. Note in the rationale "
        "how to start the Ray cluster (head + workers)."
    )
    lines.append("")
    lines.append(
        "DO NOT set per-node identity env vars in the recipe's `env:` "
        "block. Upstream's launch-cluster.sh sets these per-node via "
        "`docker run -e`, and the recipe's `env:` block is exported "
        "INSIDE the container by run-recipe.py. Setting them in `env:` "
        "either broadcasts a single wrong value to every node or — when "
        "the value references `$LOCAL_IP` — expands to empty (LOCAL_IP "
        "is not defined inside the container) and blanks out the "
        "correct value. Either path makes vLLM auto-detect the eth IP "
        "and Ray's placement group times out. The keys upstream "
        "manages and that you must NEVER include in the recipe `env:`:\n"
        "  VLLM_HOST_IP, RAY_NODE_IP_ADDRESS, RAY_OVERRIDE_NODE_IP_ADDRESS,\n"
        "  NCCL_SOCKET_IFNAME, NCCL_IB_HCA, GLOO_SOCKET_IFNAME,\n"
        "  TP_SOCKET_IFNAME, UCX_NET_DEVICES, MN_IF_NAME,\n"
        "  OMPI_MCA_btl_tcp_if_include\n"
        "Restrict `env:` to model-specific knobs (e.g. VLLM_USE_DEEP_GEMM, "
        "OMP_NUM_THREADS) and prefer leaving it empty (`env: {}`) when "
        "the upstream cluster recipes for similar models do."
    )
    return "\n".join(lines)


def build_recipe_prompt(
    info: HFModelInfo,
    caps: BoxCapabilities,
    *,
    cluster: dict | None = None,
) -> str:
    parts = [_model_block(info), "", _caps_block(caps), ""]
    if cluster:
        parts.append(_cluster_block(cluster))
        parts.append("")
    parts.append(
        "Produce a RecipeDraft as JSON with keys: "
        "`name` (slug derived from model), `model` (HF id), `args` (dict of "
        "CLI flag → value strings), `env` (dict), `description`, `rationale`.\n"
    )
    return "\n".join(parts)


def build_optimize_prompt(
    recipe: RecipeSpec,
    caps: BoxCapabilities,
    *,
    goals: list[str],
    cluster: dict | None = None,
) -> str:
    parts = [
        f"Existing recipe:\n```yaml\n"
        f"name: {recipe.name}\nmodel: {recipe.model}\n"
        f"args: {json.dumps(recipe.args)}\nenv: {json.dumps(recipe.env)}\n"
        f"```\n",
        _caps_block(caps),
        f"\nModel fact: {render_tool_call_block(recipe.model)}",
    ]
    if cluster:
        parts.append("")
        parts.append(_cluster_block(cluster))
    parts.append("")
    parts.append(f"Goals (in priority order): {', '.join(goals)}")
    parts.append("")
    parts.append(
        "Return a revised RecipeDraft (same JSON shape as recipe creation). "
        "Keep the same `name` and `model`. Explain each change in "
        "`rationale`. Reconcile the recipe's tool-call args with the "
        "Model fact above — if SUPPORTED and args don't have both "
        "--tool-call-parser and --enable-auto-tool-choice, add them; if "
        "NOT SUPPORTED, remove them; if UNKNOWN, preserve the recipe's "
        "existing tool-call args exactly as they are (never remove them "
        "on a heuristic miss)."
    )
    if cluster:
        total = cluster.get("total_gpus", 0)
        parts.append(
            f"REMINDER: target is a {len(cluster.get('nodes') or [])}-node "
            f"cluster with {total} total GPUs. The product of "
            f"--tensor-parallel-size and --pipeline-parallel-size in the "
            f"revised recipe MUST equal {total}. If the existing recipe "
            f"is undersized for the cluster (e.g. tp=1 against {total} "
            f"GPUs), upsize it; if oversized, downsize."
        )
    return "\n".join(parts) + "\n"


def build_mod_prompt(*, error_log: str, model_id: str) -> str:
    return (
        f"Model: {model_id}\n\n"
        f"Error log / failure mode:\n```\n{error_log}\n```\n\n"
        "Propose a vLLM mod (a small patch + optional shell hook) that fixes this. "
        "Return a ModDraft as JSON with keys: `name`, `target_models` (list), "
        "`files` (dict of relative-path → file-contents string; typically "
        "`patch.diff` with a unified diff and optionally `hook.sh`), "
        "`description`, `rationale`.\n"
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _FENCE.search(text)
    if not m:
        return json.loads(text)
    return json.loads(m.group(1))


def parse_recipe_draft(text: str) -> RecipeDraft:
    data = _extract_json(text)
    return RecipeDraft(**data)


def parse_mod_draft(text: str) -> ModDraft:
    data = _extract_json(text)
    return ModDraft(**data)
