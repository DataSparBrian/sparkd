"""Curated mapping that decides whether a HF model id supports tool
calling and which vLLM `--tool-call-parser` to use. This is what the
advisor's prompt embeds as a fact so Claude doesn't have to guess and
produce inconsistent configs (parser without enable-auto-tool-choice,
parser for a base/pretraining model, etc.).

Tri-state: a table hit is "supported", a base/pretraining marker is
"unsupported", everything else is "unknown" — and "unknown" must never
be rendered as an instruction to strip existing tool-call args (the
Nemotron-3 regression)."""

from sparkd.advisor.tool_calls import (
    ToolCallSupport,
    infer_tool_call_config,
    render_tool_call_block,
)


def test_qwen3_5_uses_qwen3_coder_parser():
    r = infer_tool_call_config("Qwen/Qwen3.5-122B-A10B-FP8")
    assert r == ToolCallSupport(status="supported", parser="qwen3_coder")


def test_qwen3_6_uses_qwen3_coder_parser():
    """Forward-compat: Qwen3.6 series maps to the same parser."""
    r = infer_tool_call_config("mmangkad/Qwen3.6-27B-NVFP4")
    assert r == ToolCallSupport(status="supported", parser="qwen3_coder")


def test_qwen2_5_uses_hermes_parser():
    """vLLM has no `qwen2_5` parser — Qwen2.5 instruct models use the
    hermes-format parser. The old mapping shipped invalid recipes."""
    r = infer_tool_call_config("Qwen/Qwen2.5-7B-Instruct")
    assert r == ToolCallSupport(status="supported", parser="hermes")


def test_plain_qwen3_uses_hermes_parser():
    """Plain (non-coder) Qwen3 instruct models use hermes, not the
    coder parser. More-specific rows (qwen3-coder, qwen3.5/3.6) match
    first."""
    r = infer_tool_call_config("Qwen/Qwen3-32B")
    assert r == ToolCallSupport(status="supported", parser="hermes")


def test_nemotron_3_super_uses_qwen3_coder_parser():
    """Regression: the Nemotron-3 family was missing from the table, so
    the advisor rendered 'NOT detected' and stripped a working
    tool-calling config. NVIDIA's DGX Spark guide serves Nemotron 3
    with the qwen3_coder parser."""
    r = infer_tool_call_config("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert r == ToolCallSupport(status="supported", parser="qwen3_coder")


def test_nemotron_3_nano_uses_qwen3_coder_parser():
    r = infer_tool_call_config("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
    assert r == ToolCallSupport(status="supported", parser="qwen3_coder")


def test_library_families_resolve():
    """Families present in the recipe library map to the parsers their
    curated recipes use."""
    cases = {
        "google/gemma-4-26B-A4B-it": "gemma4",
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4": "gemma4",
        "cyankiwi/GLM-4.7-Flash-AWQ-4bit": "glm47",
        "QuantTrio/MiniMax-M2-AWQ": "minimax_m2",
        "cyankiwi/MiniMax-M2.7-AWQ-4bit": "minimax_m2",
        "openai/gpt-oss-120b": "openai",
        "stepfun-ai/Step-3.7-Flash-FP8": "step3p5",
    }
    for model_id, parser in cases.items():
        r = infer_tool_call_config(model_id)
        assert r == ToolCallSupport(status="supported", parser=parser), model_id


def test_llama3_1_uses_json_parser():
    r = infer_tool_call_config("meta-llama/Llama-3.1-70B-Instruct")
    assert r == ToolCallSupport(status="supported", parser="llama3_json")


def test_mixtral_uses_mistral_parser():
    r = infer_tool_call_config("mistralai/Mixtral-8x22B-Instruct-v0.1")
    assert r == ToolCallSupport(status="supported", parser="mistral")


def test_phi4_mini_dedicated_parser():
    r = infer_tool_call_config("microsoft/Phi-4-Mini-Instruct")
    assert r == ToolCallSupport(status="supported", parser="phi4_mini_json")


def test_deepseek_v3_dedicated_parser():
    r = infer_tool_call_config("deepseek-ai/DeepSeek-V3")
    assert r == ToolCallSupport(status="supported", parser="deepseek_v3")


def test_base_model_is_positively_unsupported():
    """Base / pretraining variants don't ship a chat template — no
    tool calling possible even when the family supports it. This is a
    known negative, not an unknown."""
    r = infer_tool_call_config("Qwen/Qwen3-8B-Base")
    assert r == ToolCallSupport(status="unsupported", parser=None)
    assert r.supports is False


def test_unknown_family_is_unknown_not_unsupported():
    """A table miss proves nothing about the model — it must come back
    'unknown', never 'unsupported'."""
    r = infer_tool_call_config("some-org/random-experimental-model")
    assert r == ToolCallSupport(status="unknown", parser=None)
    assert r.supports is False


def test_render_block_when_supported_includes_both_flag_names():
    """The rendered fact must mention both flags by name so Claude
    can't accidentally set just one."""
    s = render_tool_call_block("Qwen/Qwen3.5-122B-A10B-FP8")
    assert "SUPPORTED" in s
    assert "qwen3_coder" in s
    assert "--tool-call-parser" in s
    assert "--enable-auto-tool-choice" in s
    assert "Set both or neither" in s


def test_render_block_when_unsupported_says_do_not_set():
    """For positively-unsupported models (base variants), the line tells
    Claude explicitly NOT to set the flags — preventing the inverse
    failure mode (advisor enables tool calling on a base model that
    crashes vLLM at startup)."""
    s = render_tool_call_block("Qwen/Qwen3-8B-Base")
    assert "NOT SUPPORTED" in s
    assert "Do NOT set" in s
    assert "--tool-call-parser" in s


def test_render_block_when_unknown_says_keep_existing():
    """Regression for the Nemotron-3 incident: an unknown family must
    render as 'preserve what's there', never as an instruction to
    remove tool-call args."""
    s = render_tool_call_block("some-org/random-experimental-model")
    assert "UNKNOWN" in s
    assert "KEEP" in s
    assert "remove" not in s.lower() or "never remove" in s.lower()
    assert "Do NOT set" not in s
