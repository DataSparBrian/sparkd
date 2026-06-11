from __future__ import annotations

import yaml

from sparkd.db.engine import session_scope
from sparkd.db.models import Box
from sparkd.schemas.recipe import RecipeDiff, RecipeSpec
from sparkd.services.box import BoxService
from sparkd.services.cli_flags import BOOL_FLAG_ARGS, FLAG_ALIASES
from sparkd.services.library import LibraryService
from sparkd.ssh.pool import SSHPool


class _LiteralStr(str):
    """Marker subclass — when this hits the YAML representer below, it's
    emitted as a literal-block scalar (`|`) instead of getting wrapped
    into a quoted style with line continuations. We use it for the
    `command:` field so the multi-line vLLM invocation reads naturally
    on disk and on the box."""


def _literal_str_representer(dumper: yaml.SafeDumper, data: _LiteralStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.SafeDumper.add_representer(_LiteralStr, _literal_str_representer)


class RecipeService:
    def __init__(
        self, library: LibraryService, boxes: BoxService, pool: SSHPool
    ) -> None:
        self.library = library
        self.boxes = boxes
        self.pool = pool

    async def validate(
        self,
        recipe: RecipeSpec,
        box_id: str,
        *,
        cluster: dict | None = None,
    ) -> list[str]:
        """Pre-flight checks against the *target's* GPU budget.

        For a single-box target, the budget is the box's `gpu_count`. For
        a cluster, it's the cluster's `total_gpus`. Validates:
          - tp × pp must fit the budget (was: just tp ≤ head.gpu_count,
            which silently rejected legitimate cluster recipes like
            tp=2 against a 2-node-of-1-GPU cluster because the head only
            has 1 GPU)
          - --gpu-memory-utilization is in (0, 1]
        """
        caps = await self.boxes.capabilities(box_id)
        issues: list[str] = []
        if cluster:
            gpu_budget = int(cluster.get("total_gpus", 0)) or caps.gpu_count
            budget_label = (
                f"{gpu_budget} GPUs across cluster "
                f"'{cluster.get('name', '?')}'"
            )
        else:
            gpu_budget = caps.gpu_count
            budget_label = f"{gpu_budget} GPUs on this box"

        tp_raw = recipe.args.get("--tensor-parallel-size")
        pp_raw = recipe.args.get("--pipeline-parallel-size", "1")
        tp: int | None = None
        pp: int | None = None
        if tp_raw is not None:
            try:
                tp = int(tp_raw)
            except ValueError:
                issues.append(
                    f"--tensor-parallel-size not an integer: {tp_raw!r}"
                )
        try:
            pp = int(pp_raw)
        except ValueError:
            issues.append(
                f"--pipeline-parallel-size not an integer: {pp_raw!r}"
            )
            pp = None
        if tp is not None and pp is not None:
            product = tp * pp
            if product > gpu_budget:
                issues.append(
                    f"tp×pp={product} (--tensor-parallel-size={tp}, "
                    f"--pipeline-parallel-size={pp}) exceeds {budget_label}"
                )
            elif cluster and product != gpu_budget:
                # Soft warning: leaving GPUs idle on a cluster target
                # is almost certainly a mistake.
                issues.append(
                    f"tp×pp={product} leaves {gpu_budget - product} GPU(s) "
                    f"idle on {budget_label}"
                )

        gmu_raw = recipe.args.get("--gpu-memory-utilization")
        if gmu_raw is not None:
            try:
                gmu = float(gmu_raw)
            except ValueError:
                issues.append(f"--gpu-memory-utilization not a float: {gmu_raw!r}")
            else:
                if not 0.0 < gmu <= 1.0:
                    issues.append(
                        f"--gpu-memory-utilization={gmu} must be in (0, 1]"
                    )

        # Tool-call configuration: vLLM rejects --tool-call-parser unless
        # --enable-auto-tool-choice is also set. Surface this pre-flight
        # rather than letting it get to vLLM and crash mid-startup with
        # `"auto" tool choice requires --enable-auto-tool-choice and
        # --tool-call-parser to be set`.
        tcp = recipe.args.get("--tool-call-parser")
        eatc = recipe.args.get("--enable-auto-tool-choice")
        if tcp and not _is_true(eatc):
            issues.append(
                f"--tool-call-parser={tcp!r} requires "
                "--enable-auto-tool-choice=true to also be set "
                "(vLLM's auto tool choice needs both flags or neither)"
            )

        return issues

    async def sync(
        self,
        name: str,
        box_id: str,
        *,
        extra_env: dict[str, str] | None = None,
        strip_env_keys: list[str] | None = None,
    ) -> None:
        """Push the recipe YAML to the head box.

        Normally we forward the raw on-disk YAML byte-identical so
        upstream-format recipes (with `defaults`, `command`, `container`,
        etc.) round-trip exactly and remain runnable by the box's
        ./run-recipe.sh.

        Two transformations are supported, both applied to a parsed copy
        of the YAML so the on-disk file is never mutated:

        - `extra_env`: defaults-only merge into `env:`. Keys already
          present in the recipe are left alone.
        - `strip_env_keys`: remove these keys from `env:` entirely. Used
          by the cluster launch path to drop entries that reference
          `$LOCAL_IP` (a variable upstream does not set inside the
          container) before they reach the box and break vLLM's host-IP
          discovery.
        """
        yaml_text = self.library.load_recipe_text(name, box_id=box_id)
        data = yaml.safe_load(yaml_text) or {}
        mutated = False

        # When the recipe carries a non-empty `args:` block (typical of
        # sparkd-rendered and advisor-generated recipes), regenerate the
        # `command` template so every flag in args actually reaches vLLM.
        #
        # Why: the renderer emits `args:` as informational metadata next
        # to a hand-curated `command` template that uses `{var}`
        # substitutions from `defaults`. Flags the advisor adds (like
        # `--pipeline-parallel-size`, `--quantization`,
        # `--distributed-executor-backend`, `--trust-remote-code`) live
        # only in `args` and never reach the command line — and even
        # `--tensor-parallel-size` doesn't, because the template's
        # `{tensor_parallel}` reads `defaults.tensor_parallel` (often
        # stale), not `args.--tensor-parallel-size`. Net effect: the
        # advisor's tp=2 / pp=1 / quantization recipe launches as `-tp 1`
        # with no quantization. Bridge here so args is the source of
        # truth.
        #
        # Recipes with empty `args:` (upstream-format with curated
        # commands) pass through unchanged.
        args_dict = data.get("args") or {}
        top_model = data.get("model")
        if isinstance(args_dict, dict) and args_dict:
            # A recipe can carry BOTH a hand-curated `command` and an
            # `args:` block that have drifted apart — e.g. tool-call
            # flags (`--enable-auto-tool-choice`, `--tool-call-parser`)
            # present only in the curated command. Regenerating purely
            # from args would silently drop those flags from the actual
            # vllm invocation. Merge command-only flags into args first;
            # args wins on conflicts. `{placeholder}` values from the
            # curated command are kept verbatim — upstream's str.format
            # against `defaults` resolves them on the box.
            curated = data.get("command")
            if isinstance(curated, str) and curated.strip():
                merged = dict(args_dict)
                for k, v in _parse_command_flags(curated).items():
                    merged.setdefault(k, v)
                args_dict = merged
                data["args"] = merged
            data["command"] = _LiteralStr(_command_from_args(args_dict))
            # Ensure {model} resolves: defaults.model mirrors top-level.
            if isinstance(top_model, str) and top_model:
                defaults = data.get("defaults") or {}
                if isinstance(defaults, dict) and "model" not in defaults:
                    defaults["model"] = top_model
                    data["defaults"] = defaults
            mutated = True
        else:
            # No args → fall back to the legacy mirror-into-defaults fix
            # for upstream-format recipes that reference {model} but lack
            # `defaults.model`.
            command = data.get("command") or ""
            if (
                isinstance(command, str)
                and "{model}" in command
                and isinstance(top_model, str)
                and top_model
            ):
                defaults = data.get("defaults") or {}
                if isinstance(defaults, dict) and "model" not in defaults:
                    defaults["model"] = top_model
                    data["defaults"] = defaults
                    mutated = True

        # Strip + merge env entries if requested.
        if extra_env or strip_env_keys:
            env = data.setdefault("env", {}) or {}
            data["env"] = env
            if strip_env_keys:
                before = len(env)
                for k in strip_env_keys:
                    env.pop(k, None)
                if len(env) != before:
                    mutated = True
            if extra_env:
                for k, v in extra_env.items():
                    if k not in env:
                        env[k] = v
                        mutated = True

        if mutated:
            yaml_text = yaml.safe_dump(
                data, sort_keys=False, default_flow_style=False
            )
        if not yaml_text.endswith("\n"):
            yaml_text += "\n"
        box = await self.boxes.get(box_id)
        async with session_scope() as s:
            row = await s.get(Box, box_id)
            target = self.boxes._target_for(row)
        # Don't shlex.quote repo_path or name here:
        # - quoting `~/spark-vllm-docker` produces `'~/...'` which suppresses
        #   tilde expansion, dropping files into a literal `~` directory.
        # - name is already validated by LibraryService._NAME_RE; repo_path is
        #   user-set config (the user is the only one who can hurt themselves).
        await self.pool.run(target, f"mkdir -p {box.repo_path}/recipes")
        cmd = (
            f"cat > {box.repo_path}/recipes/{name}.yaml "
            f"<<'SPARKD_EOF'\n{yaml_text}SPARKD_EOF\n"
        )
        await self.pool.run(target, cmd)

    def diff(self, a: RecipeSpec, b: RecipeSpec) -> RecipeDiff:  # type: ignore[override]
        return _diff_specs(a, b)


# ---------- module-level helpers ----------


def _is_true(v: object) -> bool:
    """Liberal truthy check for recipe arg values, which arrive as strings
    from YAML. Used by the validator to recognize `--enable-auto-tool-choice:
    true` (or `'true'`, or yaml-bool `True`) as actually enabled."""
    if v is True:
        return True
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return False


# Re-exported from cli_flags (shared with services.library, which this
# module can't import from without a cycle).
_BOOL_FLAG_ARGS = BOOL_FLAG_ARGS
_FLAG_ALIASES = FLAG_ALIASES


def _parse_command_flags(command: str) -> dict[str, str]:
    """Extract `--flag [value]` pairs from a hand-curated command template.

    Used by sync() to preserve flags that exist only in the curated
    `command` when the command is regenerated from `args`. Handles
    backslash line continuations, bare boolean flags (mapped to "true"
    for known boolean args, "" otherwise — both render back as a bare
    flag in _command_from_args), and the short aliases in _FLAG_ALIASES.
    `{placeholder}` values are kept verbatim. Non-flag tokens (the
    `vllm serve <model>` prefix) are skipped.
    """
    tokens = [t for t in command.replace("\\\n", " ").split() if t != "\\"]
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        flag = _FLAG_ALIASES.get(tok, tok) if tok.startswith("-") else ""
        if not flag.startswith("--"):
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and not nxt.startswith("--") and nxt not in _FLAG_ALIASES:
            flags[flag] = nxt
            i += 2
        else:
            flags[flag] = "true" if flag in _BOOL_FLAG_ARGS else ""
            i += 1
    return flags


def _command_from_args(args: dict) -> str:
    """Render a vLLM serve command line from a recipe's args dict.

    Returns a multi-line bash command (literal block scalar style — no
    `|` prefix; the YAML serializer will pick the right form). The model
    is templated as `{model}` so upstream's str.format substitution
    against `defaults.model` keeps working — same convention as upstream
    recipes use for the model id.
    """
    parts = ["vllm serve {model}"]
    for k, v in args.items():
        v_str = str(v) if v is not None else ""
        # Boolean-style: emit just the flag for known boolean args when
        # the value is true-ish; skip entirely when explicitly false.
        if k in _BOOL_FLAG_ARGS:
            if v_str.lower() in ("true", "1", "yes", ""):
                parts.append(k)
            # else: false/0/no → omit the flag
            continue
        if v_str == "":
            # Bare flag (unknown but no value) — emit just the flag.
            parts.append(k)
        else:
            parts.append(f"{k} {v_str}")
    return " \\\n  ".join(parts) + "\n"


def _diff_specs(a: RecipeSpec, b: RecipeSpec) -> RecipeDiff:
    added = {k: v for k, v in b.args.items() if k not in a.args}
    removed = {k: v for k, v in a.args.items() if k not in b.args}
    changed = {
        k: (a.args[k], b.args[k])
        for k in a.args.keys() & b.args.keys()
        if a.args[k] != b.args[k]
    }
    return RecipeDiff(name=a.name, added=added, removed=removed, changed=changed)
