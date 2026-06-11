"""Shared vLLM CLI-flag knowledge used by both the recipe sync path
(services.recipe) and the library load path (services.library).

Lives in its own module because services.recipe imports services.library
— neither can host constants the other needs without a cycle.
"""

from __future__ import annotations

# Args whose value is a flag-only boolean — emitting `--trust-remote-code true`
# breaks vLLM (it reads "true" as a positional). For these, when the recipe's
# args dict has the value "true"/"True"/"" we emit only the flag.
BOOL_FLAG_ARGS = frozenset(
    {
        "--trust-remote-code",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--disable-log-stats",
        "--disable-log-requests",
    }
)

# Short-form aliases that appear in hand-curated upstream commands.
FLAG_ALIASES = {
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
}


def canonical_flag(flag: str) -> str:
    """Map a short-form flag to its long form; pass everything else through."""
    return FLAG_ALIASES.get(flag, flag)
