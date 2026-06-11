"""Catalog of supported AI providers and their recommended current models.

Hardcoded so the UI can render dropdowns without an extra round-trip. The
'models' lists are seed suggestions — users can override with any string the
provider actually serves (especially relevant for the local-vllm provider,
where the 'model' is whatever was passed to `vllm serve`).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Single place to bump when Anthropic ships a new flagship. Used as the
# adapter default and the legacy /setup route's seed model.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


@dataclass
class ProviderDef:
    id: str
    label: str
    family: str  # "anthropic" or "openai_compat"
    requires_key: bool = True
    default_base_url: str | None = None
    base_url_editable: bool = False
    models: list[str] = field(default_factory=list)
    notes: str = ""


PROVIDERS: list[ProviderDef] = [
    ProviderDef(
        id="anthropic",
        label="Anthropic (Claude)",
        family="anthropic",
        models=[
            DEFAULT_ANTHROPIC_MODEL,
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        notes="Native Messages API with prompt caching.",
    ),
    ProviderDef(
        id="openai",
        label="OpenAI",
        family="openai_compat",
        default_base_url="https://api.openai.com/v1",
        models=[
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "gpt-4-turbo",
        ],
    ),
    ProviderDef(
        id="gemini",
        label="Google Gemini",
        family="openai_compat",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        models=[
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        notes="Uses Google's OpenAI-compatible endpoint.",
    ),
    ProviderDef(
        id="mistral",
        label="Mistral",
        family="openai_compat",
        default_base_url="https://api.mistral.ai/v1",
        models=[
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "codestral-latest",
        ],
    ),
    ProviderDef(
        id="groq",
        label="Groq",
        family="openai_compat",
        default_base_url="https://api.groq.com/openai/v1",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
    ),
    ProviderDef(
        id="openrouter",
        label="OpenRouter",
        family="openai_compat",
        default_base_url="https://openrouter.ai/api/v1",
        models=[
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-exp",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        notes="Single key, many upstream models.",
    ),
    ProviderDef(
        id="together",
        label="Together AI",
        family="openai_compat",
        default_base_url="https://api.together.xyz/v1",
        models=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ],
    ),
    ProviderDef(
        id="vllm",
        label="Local vLLM",
        family="openai_compat",
        requires_key=False,
        default_base_url="http://localhost:8000/v1",
        base_url_editable=True,
        models=[],  # user-defined; they pass the model name to `vllm serve`
        notes="Point at any vLLM (or other OpenAI-compatible) server you run yourself.",
    ),
]


def get_provider(provider_id: str) -> ProviderDef | None:
    for p in PROVIDERS:
        if p.id == provider_id:
            return p
    return None
