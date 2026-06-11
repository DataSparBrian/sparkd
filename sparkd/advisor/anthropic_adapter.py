from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic

from sparkd.advisor.prompts import (
    SYSTEM_PROMPT,
    build_mod_prompt,
    build_optimize_prompt,
    build_recipe_prompt,
)
from sparkd.advisor.providers import DEFAULT_ANTHROPIC_MODEL
from sparkd.schemas.advisor import AdvisorMessage
from sparkd.schemas.box import BoxCapabilities
from sparkd.schemas.hf import HFModelInfo
from sparkd.schemas.recipe import RecipeSpec


@dataclass
class AdvisorChunk:
    delta: str
    input_tokens: int = 0
    output_tokens: int = 0
    final: bool = False


class AdvisorPort(Protocol):
    def stream_recipe(
        self,
        info: HFModelInfo,
        caps: BoxCapabilities,
        history: list[AdvisorMessage],
        *,
        cluster: dict | None = ...,
    ) -> AsyncIterator[AdvisorChunk]: ...

    def stream_optimize(
        self,
        recipe: RecipeSpec,
        caps: BoxCapabilities,
        goals: list[str],
        history: list[AdvisorMessage],
        *,
        cluster: dict | None = ...,
    ) -> AsyncIterator[AdvisorChunk]: ...

    def stream_mod(
        self, error_log: str, model_id: str, history: list[AdvisorMessage]
    ) -> AsyncIterator[AdvisorChunk]: ...


class AnthropicAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def _stream(
        self, system: str, user: str, history: list[AdvisorMessage]
    ) -> AsyncIterator[AdvisorChunk]:
        messages: list[dict] = []
        for m in history:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user})
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = getattr(event.delta, "text", "") or ""
                    if delta:
                        yield AdvisorChunk(delta=delta)
            final = await stream.get_final_message()
        usage = getattr(final, "usage", None)
        if usage is not None:
            yield AdvisorChunk(
                delta="",
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                final=True,
            )
        else:
            yield AdvisorChunk(delta="", final=True)

    async def stream_recipe(
        self,
        info: HFModelInfo,
        caps: BoxCapabilities,
        history: list[AdvisorMessage],
        *,
        cluster: dict | None = None,
    ) -> AsyncIterator[AdvisorChunk]:
        prompt = build_recipe_prompt(info, caps, cluster=cluster)
        async for c in self._stream(SYSTEM_PROMPT, prompt, history):
            yield c

    async def stream_optimize(
        self,
        recipe: RecipeSpec,
        caps: BoxCapabilities,
        goals: list[str],
        history: list[AdvisorMessage],
        *,
        cluster: dict | None = None,
    ) -> AsyncIterator[AdvisorChunk]:
        prompt = build_optimize_prompt(
            recipe, caps, goals=goals, cluster=cluster
        )
        async for c in self._stream(SYSTEM_PROMPT, prompt, history):
            yield c

    async def stream_mod(
        self, error_log: str, model_id: str, history: list[AdvisorMessage]
    ) -> AsyncIterator[AdvisorChunk]:
        prompt = build_mod_prompt(error_log=error_log, model_id=model_id)
        async for c in self._stream(SYSTEM_PROMPT, prompt, history):
            yield c
