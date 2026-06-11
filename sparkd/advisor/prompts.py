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
    # Missing facts (HF fetch failed / 404) default to 0 in HFModelInfo.
    # Never render those as literal values — "Parameters: 0.0 B" invites
    # the model to either take it at face value or silently guess. Mark
    # them unknown and demand the assumption be surfaced in `rationale`.
    params = (
        f"{info.parameters_b} B"
        if info.parameters_b
        else "unknown — do NOT guess a size; state the assumption you "
        "make in `rationale`"
    )
    ctx = (
        str(info.context_length)
        if info.context_length
        else "unknown — choose a conservative --max-model-len and flag "
        "it in `rationale`"
    )
    return (
        f"Hugging Face model facts:\n"
        f"- ID: {info.id}\n"
        f"- Architecture: {info.architecture or 'unknown'}\n"
        f"- Parameters: {params}\n"
        f"- Context length: {ctx}\n"
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
        f"PARALLELISM CONSTRAINTS: --tensor-parallel-size × "
        f"--pipeline-parallel-size must not exceed the cluster's total "
        f"GPU count of **{total_gpus}** (= {n_nodes} nodes × "
        f"{gpus_per_node} GPU/node), and --tensor-parallel-size must "
        "evenly divide the model's attention-head count. Default to "
        f"using ALL GPUs (tp × pp = {total_gpus}) — this cluster exists "
        "to serve this model, and leaving GPUs idle is almost always a "
        "mistake. Sizing below total is valid only when a real "
        "constraint forces it (head count not divisible by the GPU "
        "count, MoE expert layout, per-stage memory shape) — if you do, "
        "say why in `rationale`."
    )
    lines.append("")
    lines.append(
        f"LAYOUT TRADE-OFF: every node here has {gpus_per_node} GPU(s), "
        "so tensor-parallel groups larger than one node communicate "
        f"over the inter-node link. tp = {total_gpus}, pp = 1 gives "
        "each stage the full aggregate VRAM and is what NVIDIA's Spark "
        "playbooks use at small node counts, but all-reduce traffic "
        f"crosses the network every layer. pp = {n_nodes} with tp = "
        f"{gpus_per_node} keeps tensor traffic on-node and only sends "
        "activations between stages, at the cost of pipeline bubbles. "
        "Prefer cross-node tp for 2-node clusters and latency-sensitive "
        "serving; weigh pipeline stages as node count grows or when "
        "the model is interconnect-bound. State the choice and why in "
        "`rationale`."
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
            f"cluster with {total} total GPUs. Size the revised recipe "
            f"for the cluster: tp × pp should use all {total} GPUs "
            "unless a divisibility or memory constraint forces fewer "
            "(explain in `rationale`). If the existing recipe is sized "
            f"for a single box (e.g. tp=1 against {total} GPUs), upsize "
            f"it; if tp × pp exceeds {total}, downsize."
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
