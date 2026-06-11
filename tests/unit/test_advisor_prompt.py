from datetime import datetime, timezone

from sparkd.advisor.prompts import (
    SYSTEM_PROMPT,
    build_mod_prompt,
    build_optimize_prompt,
    build_recipe_prompt,
    parse_mod_draft,
    parse_recipe_draft,
)
from sparkd.schemas.box import BoxCapabilities
from sparkd.schemas.hf import HFModelInfo
from sparkd.schemas.recipe import RecipeSpec


def _caps() -> BoxCapabilities:
    return BoxCapabilities(
        gpu_count=2,
        gpu_model="NVIDIA GB10",
        vram_per_gpu_gb=96,
        captured_at=datetime.now(timezone.utc),
    )


def _info() -> HFModelInfo:
    return HFModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        architecture="LlamaForCausalLM",
        parameters_b=8.0,
        context_length=131072,
        supported_dtypes=["bf16"],
    )


def test_recipe_prompt_includes_facts():
    p = build_recipe_prompt(_info(), _caps())
    assert "GB10" in p
    assert "Llama-3.1-8B" in p
    assert "131072" in p


def test_optimize_prompt_carries_existing_recipe():
    r = RecipeSpec(name="r", model="m", args={"--tp": "1"})
    p = build_optimize_prompt(r, _caps(), goals=["throughput"])
    assert "--tp" in p
    assert "throughput" in p


def _cluster(n_nodes: int, gpus_per_node: int = 1, vram_gb: int = 128) -> dict:
    return {
        "name": "alpha",
        "nodes": [
            {
                "name": f"n{i}",
                "host": f"10.0.0.{i}",
                "gpu_count": gpus_per_node,
                "gpu_model": "NVIDIA GB10",
                "vram_gb": vram_gb,
                "ib": "mlx5_0",
            }
            for i in range(n_nodes)
        ],
        "total_gpus": n_nodes * gpus_per_node,
        "total_vram_gb": n_nodes * gpus_per_node * vram_gb,
    }


def test_recipe_prompt_with_cluster_pins_tp_pp_to_total_gpus():
    """When a cluster context is provided, the prompt must explicitly
    tell the model that --tensor-parallel-size × --pipeline-parallel-size
    has to equal the cluster's total GPU count. Without this the AI keeps
    sizing recipes for a single node."""
    p = build_recipe_prompt(_info(), _caps(), cluster=_cluster(n_nodes=3))
    # Hard binding stated.
    assert "tensor-parallel-size × --pipeline-parallel-size MUST equal" in p
    # The actual number is in the prompt.
    assert "**3**" in p
    # Preferred layout (tp = total_gpus, pp = 1) is suggested.
    assert "tp = 3, pp = 1" in p


def test_optimize_prompt_with_cluster_includes_topology_and_tp_constraint():
    """Optimize must ALSO get the cluster context — without it, an
    Optimize click against a cluster target silently returns advice
    sized for a single box."""
    r = RecipeSpec(name="r", model="m", args={"--tensor-parallel-size": "1"})
    p = build_optimize_prompt(
        r, _caps(), goals=["throughput"], cluster=_cluster(n_nodes=2)
    )
    assert "Multi-node cluster topology" in p
    assert "Total GPUs across cluster: 2" in p
    # Reminder line tells Claude to upsize/downsize against total_gpus.
    assert (
        "tensor-parallel-size and --pipeline-parallel-size in the "
        "revised recipe MUST equal 2"
    ) in p


def test_optimize_prompt_without_cluster_unchanged_behavior():
    """No cluster → no cluster section, no reminder — same as before."""
    r = RecipeSpec(name="r", model="m", args={"--tp": "1"})
    p = build_optimize_prompt(r, _caps(), goals=["throughput"])
    assert "Multi-node cluster topology" not in p
    assert "MUST equal" not in p


def test_recipe_prompt_embeds_tool_call_fact_for_supported_model():
    """When the model id matches a curated parser pattern, the recipe
    prompt embeds it as a concrete fact — the advisor doesn't have to
    guess. Same approach prevents the "set parser without enable-auto"
    failure mode that triggered this whole feature."""
    qwen_info = HFModelInfo(
        id="Qwen/Qwen3.5-122B-A10B-FP8",
        architecture="Qwen3MoeForCausalLM",
        parameters_b=122.0,
        context_length=262144,
    )
    p = build_recipe_prompt(qwen_info, _caps())
    assert "Tool calling: SUPPORTED" in p
    assert "qwen3_coder" in p
    assert "--enable-auto-tool-choice" in p


def test_recipe_prompt_embeds_no_tool_call_for_base_model():
    base_info = HFModelInfo(
        id="Qwen/Qwen3-8B-Base",
        architecture="Qwen3ForCausalLM",
        parameters_b=8.0,
        context_length=32768,
    )
    p = build_recipe_prompt(base_info, _caps())
    assert "Tool calling: NOT SUPPORTED" in p
    assert "Do NOT set" in p


def test_optimize_prompt_unknown_family_says_preserve():
    """Regression for the Nemotron-3 incident: when the family isn't in
    the parser table, the optimize prompt must instruct the advisor to
    preserve existing tool-call args, not remove them."""
    r = RecipeSpec(
        name="r",
        model="some-org/random-experimental-model",
        args={
            "--tool-call-parser": "custom",
            "--enable-auto-tool-choice": "true",
        },
    )
    p = build_optimize_prompt(r, _caps(), goals=["throughput"])
    assert "Tool calling: UNKNOWN" in p
    assert "KEEP" in p
    assert "if UNKNOWN, preserve" in p


def test_optimize_prompt_embeds_tool_call_fact():
    """The optimize prompt also gets the tool-call fact, so when a user
    asks to optimize a recipe whose tool-call config doesn't match the
    model's actual support, Claude sees ground truth and can fix it."""
    r = RecipeSpec(
        name="r",
        model="meta-llama/Llama-3.1-70B-Instruct",
        args={"--tp": "1"},
    )
    p = build_optimize_prompt(r, _caps(), goals=["throughput"])
    assert "Tool calling: SUPPORTED" in p
    assert "llama3_json" in p
    # The reconcile instruction tells Claude to fix mismatches.
    assert "Reconcile the recipe's tool-call args" in p


def test_mod_prompt_carries_error_log():
    p = build_mod_prompt(error_log="ImportError: foo", model_id="x")
    assert "ImportError: foo" in p


def test_parse_recipe_draft_from_json_block():
    text = (
        "Here is the recipe.\n"
        "```json\n"
        '{"name":"r","model":"m","args":{"--tp":"2"},'
        '"env":{},"description":"d","rationale":"r"}\n'
        "```\n"
    )
    draft = parse_recipe_draft(text)
    assert draft.name == "r"
    assert draft.args["--tp"] == "2"


def test_parse_mod_draft_from_json_block():
    text = (
        "```json\n"
        '{"name":"m1","target_models":["llama"],'
        '"files":{"patch.diff":"--- a\\n+++ b\\n"},'
        '"description":"d","rationale":"r"}\n'
        "```"
    )
    d = parse_mod_draft(text)
    assert d.name == "m1"
    assert "patch.diff" in d.files


def test_system_prompt_describes_role():
    assert "DGX Spark" in SYSTEM_PROMPT
