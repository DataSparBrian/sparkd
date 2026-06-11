"""Determine whether a Hugging Face model supports tool calling, and which
vLLM `--tool-call-parser` to use for it.

Pure inference from the model id — no network calls. Substring-matched
against a curated mapping of family → parser. The advisor surfaces this
as a fact in its prompt so Claude (or any LLM advisor) doesn't have to
guess and produce inconsistent configurations like
`--tool-call-parser=qwen3_coder` without `--enable-auto-tool-choice`.

The result is tri-state. A table hit means "supported, use this parser".
A base/pretraining marker means "unsupported" (no chat template, so tool
calling is genuinely impossible). Anything else is "unknown" — the table
can only ever prove presence, never absence, so a miss must NOT be
treated as evidence against tool calling. The prompt renders "unknown"
as an instruction to preserve whatever tool-call args the recipe already
has (see the Nemotron-3 incident: a missing row caused the advisor to
strip a working tool-calling config).

Updating the mapping: add a new (substring, parser) tuple to
_PARSER_PATTERNS. Order matters — more-specific substrings first
(e.g. `qwen3-coder` before `qwen3`). Patterns are case-insensitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ToolCallStatus = Literal["supported", "unsupported", "unknown"]


@dataclass(frozen=True)
class ToolCallSupport:
    """Result of `infer_tool_call_config(model_id)`.

    `status` is "supported" when the model family is known to ship a
    chat template + tokenizer config that vLLM's `--tool-call-parser`
    can parse; "unsupported" when the variant is known NOT to (base /
    pretraining checkpoints); "unknown" when the family simply isn't in
    the table. `parser` is the value to pass to `--tool-call-parser`
    when status is "supported"; `None` otherwise.
    """

    status: ToolCallStatus
    parser: str | None

    @property
    def supports(self) -> bool:
        return self.status == "supported"


# (substring, parser) — substrings are matched against the lowercased
# model id (org/name). Listed more-specific first because the loop
# returns on first match. Updates: when vLLM adds a new parser, add a
# row here for the matching family. Parser names must exist in the vLLM
# version deployed on the boxes — when in doubt, check a known-good
# recipe in the library for the same family.
_PARSER_PATTERNS: list[tuple[str, str]] = [
    # Qwen — the coder line and the 3.5/3.6 series use the coder parser
    # (newer vLLM also ships it as `qwen3_xml`); plain Qwen3 and Qwen2.5
    # instruct models use the hermes-format parser. NOTE: vLLM has no
    # `qwen2_5` parser — that was a bug here that shipped invalid
    # recipes.
    ("qwen3-coder", "qwen3_coder"),
    ("qwen3.5", "qwen3_coder"),
    ("qwen3.6", "qwen3_coder"),
    ("qwen3", "hermes"),
    ("qwen2.5", "hermes"),
    # NVIDIA Nemotron 3 (Nano/Super/Ultra) — NVIDIA's DGX Spark
    # deployment guide serves these with the qwen3_coder parser
    # (alongside --reasoning-parser nemotron_v3, which is a separate
    # flag the recipe carries itself).
    ("nemotron-3", "qwen3_coder"),
    # Gemma 4 / DiffusionGemma.
    ("diffusiongemma", "gemma4"),
    ("gemma-4", "gemma4"),
    # GLM 4.7.
    ("glm-4.7", "glm47"),
    # MiniMax M2 line (matches M2, M2.5, M2.7).
    ("minimax-m2", "minimax_m2"),
    # OpenAI OSS models.
    ("gpt-oss", "openai"),
    # StepFun Step 3.7.
    ("step-3.7", "step3p5"),
    # Llama 3.x — tool calling via JSON-mode parser.
    ("llama-3.1", "llama3_json"),
    ("llama-3.2", "llama3_json"),
    ("llama-3.3", "llama3_json"),
    ("llama-4", "llama3_json"),  # forward-compat best guess
    # Mistral / Mixtral — same parser.
    ("mistral-large", "mistral"),
    ("mistral-7b-instruct-v0.3", "mistral"),
    ("mistral-nemo", "mistral"),
    ("mixtral", "mistral"),
    # NousResearch Hermes fine-tunes.
    ("hermes-2-pro", "hermes"),
    ("hermes-3", "hermes"),
    # InternLM 2.5+
    ("internlm2_5", "internlm"),
    ("internlm3", "internlm"),
    # IBM Granite 3+
    ("granite-3", "granite"),
    # Microsoft Phi-4 Mini
    ("phi-4-mini", "phi4_mini_json"),
    # DeepSeek
    ("deepseek-v3", "deepseek_v3"),
    ("deepseek-r1", "deepseek_v3"),  # R1 shares V3 tool-call format
]

# Markers in the model id that indicate a base / pretraining-only model
# — even if the family supports tool calling, the base variant doesn't
# have the instruction-tuned chat template needed for it.
_BASE_MARKERS: tuple[str, ...] = (
    "-base",
    "-pretrain",
    "/pythia-",
    "-completion",
)


def infer_tool_call_config(model_id: str) -> ToolCallSupport:
    """Infer tool-call support from a HF model id. Returns ToolCallSupport.

    Heuristic, by design — we keep this dependency-free (no HF fetch) so
    it runs in the prompt builder without latency. A table miss returns
    "unknown", never "unsupported": the advisor must not strip existing
    tool-call args just because a family hasn't been added here yet.
    Only base/pretraining variants are positively "unsupported". False
    positives mean the advisor proposes a parser that vLLM might reject;
    the curated mapping should be kept current against the deployed
    vLLM's parsers.
    """
    lower = model_id.lower()
    # Base/pretraining variants never support tool calling, even when
    # the family does.
    if any(m in lower for m in _BASE_MARKERS):
        return ToolCallSupport(status="unsupported", parser=None)
    for substr, parser in _PARSER_PATTERNS:
        if substr in lower:
            return ToolCallSupport(status="supported", parser=parser)
    return ToolCallSupport(status="unknown", parser=None)


def render_tool_call_block(model_id: str) -> str:
    """Format the inference result as a single line for embedding in
    advisor prompts. "supported" and "unsupported" are stated as
    concrete facts; "unknown" explicitly tells the advisor to preserve
    the recipe's existing tool-call args — a missing table row proves
    nothing about the model."""
    tc = infer_tool_call_config(model_id)
    if tc.status == "supported":
        return (
            f"Tool calling: SUPPORTED. Use "
            f'--tool-call-parser: "{tc.parser}" '
            'AND --enable-auto-tool-choice: "true". Set both or neither.'
        )
    if tc.status == "unsupported":
        return (
            "Tool calling: NOT SUPPORTED (base/pretraining variant — no "
            "instruction-tuned chat template). Do NOT set "
            "--tool-call-parser or --enable-auto-tool-choice; remove "
            "them if present."
        )
    return (
        "Tool calling: UNKNOWN — this model family is not in sparkd's "
        "parser table, which proves nothing about the model itself. If "
        "the recipe already sets --tool-call-parser and "
        "--enable-auto-tool-choice, KEEP them exactly as they are. Do "
        "not add tool-call args on your own; note the uncertainty in "
        "`rationale` instead."
    )
