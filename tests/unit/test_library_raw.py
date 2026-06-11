"""save_recipe_raw round-trips upstream-format YAML byte-identical."""

import pytest

from sparkd.errors import ValidationError
from sparkd.services.library import LibraryService


UPSTREAM_YAML = """recipe_version: 1
name: qwen3.5-35b-a3b-fp8
description: Qwen 3.5 35B A3B FP8 quantization
model: zai-org/Qwen3.5-35B-A3B-FP8
container: vllm/vllm-openai:latest
mods:
  - mods/qwen-marlin-fix
  - mods/fp8-kv
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.92
  max_model_len: 32768
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
command: |
  vllm serve {model}
    --tensor-parallel-size {tensor_parallel}
    --gpu-memory-utilization {gpu_memory_utilization}
"""


@pytest.fixture
def lib(sparkd_home):
    return LibraryService()


def test_save_raw_preserves_yaml_byte_for_byte(lib):
    spec = lib.save_recipe_raw("qwen3.5-35b-a3b-fp8", UPSTREAM_YAML)
    assert spec.name == "qwen3.5-35b-a3b-fp8"
    assert spec.model == "zai-org/Qwen3.5-35B-A3B-FP8"
    text = lib.load_recipe_text("qwen3.5-35b-a3b-fp8")
    assert text == UPSTREAM_YAML


def test_save_raw_preserves_name_field_in_yaml_but_view_uses_filename(lib):
    yaml_text = "name: not-the-filename\nmodel: x/y\n"
    spec = lib.save_recipe_raw("real-name", yaml_text)
    # Parsed view: filename wins.
    assert spec.name == "real-name"
    # On-disk: YAML untouched.
    assert lib.load_recipe_text("real-name") == yaml_text


def test_save_raw_rejects_missing_model(lib):
    with pytest.raises(ValidationError):
        lib.save_recipe_raw("x", "name: x\ndescription: no model\n")


def test_save_raw_rejects_non_mapping(lib):
    with pytest.raises(ValidationError):
        lib.save_recipe_raw("x", "- a\n- b\n")


def test_has_recipe(lib):
    assert lib.has_recipe("nope") is False
    lib.save_recipe_raw("yep", "name: yep\nmodel: m\n")
    assert lib.has_recipe("yep") is True


def test_save_raw_accepts_null_env(lib):
    """Some upstream YAMLs have `env:` with no value (parsed as None)."""
    spec = lib.save_recipe_raw("with-null-env", "name: x\nmodel: a/b\nenv:\n")
    assert spec.env == {}
    assert lib.has_recipe("with-null-env")


def test_save_raw_coerces_int_env_values(lib):
    """Upstream sets env vars as ints (VLLM_MARLIN_USE_ATOMIC_ADD: 1)."""
    yaml_text = "name: x\nmodel: a/b\nenv:\n  FOO: 1\n  BAR: true\n"
    spec = lib.save_recipe_raw("with-int-env", yaml_text)
    assert spec.env == {"FOO": "1", "BAR": "True"}
    # On-disk YAML preserved verbatim — int stays int.
    assert lib.load_recipe_text("with-int-env") == yaml_text


def test_save_raw_does_not_leave_orphan_on_validation_failure(lib):
    """Failed schema validation must not write a partial file."""
    with pytest.raises(ValidationError):
        # mods must be a list, not a string — RecipeSpec rejects.
        lib.save_recipe_raw(
            "mods-wrong", "name: m\nmodel: a/b\nmods: 'not-a-list'\n"
        )
    assert not lib.has_recipe("mods-wrong")


def test_load_and_list_use_filename_as_canonical_name(lib):
    """The slug is the identifier across the API, even if the YAML body has
    a different (pretty) `name:` field — common in upstream recipes."""
    yaml_text = "name: OpenAI GPT-OSS 120B\nmodel: openai/gpt-oss-120b\n"
    lib.save_recipe_raw("openai-gpt-oss-120b", yaml_text)
    spec = lib.load_recipe("openai-gpt-oss-120b")
    assert spec.name == "openai-gpt-oss-120b"
    assert spec.model == "openai/gpt-oss-120b"
    listed = {r.name for r in lib.list_recipes()}
    assert "openai-gpt-oss-120b" in listed
    # On-disk YAML preserves the pretty internal name verbatim.
    assert "OpenAI GPT-OSS 120B" in lib.load_recipe_text("openai-gpt-oss-120b")


def test_update_recipe_preserves_internal_yaml_name(lib):
    """Form-based PUT shouldn't overwrite the YAML's pretty `name:` field
    with the slug — that would erase the upstream display name on first save."""
    from sparkd.schemas.recipe import RecipeSpec

    yaml_text = "name: OpenAI GPT-OSS 120B\nmodel: org/m\nargs: {--tp: '1'}\n"
    lib.save_recipe_raw("openai-gpt-oss-120b", yaml_text)
    # Form save with the slug as name; user edited args.
    lib.update_recipe(
        RecipeSpec(
            name="openai-gpt-oss-120b",
            model="org/m",
            args={"--tp": "2"},
        )
    )
    text = lib.load_recipe_text("openai-gpt-oss-120b")
    assert "OpenAI GPT-OSS 120B" in text  # pretty name preserved
    assert "'2'" in text or "tp: 2" in text  # args got updated


def test_load_extracts_args_from_command_template(lib):
    """Upstream recipes put flags in a templated `command:` with `{var}`
    references resolved from `defaults:`. Surface those as `args` at load
    time so the form isn't empty for synced recipes."""
    yaml_text = (
        'recipe_version: "1"\n'
        "name: q\n"
        "description: d\n"
        "model: org/m\n"
        "container: vllm-node\n"
        "mods: []\n"
        "defaults:\n"
        "  port: 8000\n"
        "  host: 0.0.0.0\n"
        "  tensor_parallel: 2\n"
        "  gpu_memory_utilization: 0.7\n"
        "  max_model_len: 131072\n"
        "env: {}\n"
        "command: |\n"
        "  vllm serve org/m \\\n"
        "    --enable-auto-tool-choice \\\n"
        "    --tool-call-parser qwen3_coder \\\n"
        "    --gpu-memory-utilization {gpu_memory_utilization} \\\n"
        "    --host {host} \\\n"
        "    --port {port} \\\n"
        "    --max-model-len {max_model_len} \\\n"
        "    -tp {tensor_parallel}\n"
    )
    lib.save_recipe_raw("q", yaml_text)
    spec = lib.load_recipe("q")
    assert spec.args["--gpu-memory-utilization"] == "0.7"
    assert spec.args["--port"] == "8000"
    assert spec.args["--max-model-len"] == "131072"
    assert spec.args["-tp"] == "2"
    assert spec.args["--tool-call-parser"] == "qwen3_coder"
    # Known boolean flags derive as "true" so validate()'s _is_true and
    # the form view read them as enabled ("" parses as false and caused
    # spurious "parser without --enable-auto-tool-choice" warnings).
    assert spec.args["--enable-auto-tool-choice"] == "true"


def test_load_merges_command_only_flags_into_existing_args(lib):
    """A recipe whose `args:` drifted from its curated `command:` (the
    Nemotron-3 case: tool-call flags only in command) must surface the
    command-only flags in the parsed args view — otherwise the advisor
    optimizes against a partial config. Existing args win per-key."""
    yaml_text = (
        "name: r\n"
        "model: org/m\n"
        "defaults:\n"
        "  tensor_parallel: 2\n"
        "args:\n"
        "  --my-flag: hello\n"
        "  --port: '9999'\n"
        "command: |\n"
        "  vllm serve org/m \\\n"
        "    --port 9000 \\\n"
        "    --enable-auto-tool-choice \\\n"
        "    --tool-call-parser qwen3_coder \\\n"
        "    -tp {tensor_parallel}\n"
    )
    lib.save_recipe_raw("r", yaml_text)
    spec = lib.load_recipe("r")
    # Existing args respected per-key.
    assert spec.args["--my-flag"] == "hello"
    assert spec.args["--port"] == "9999"
    # Command-only flags merged in, short aliases canonicalized.
    assert spec.args["--tool-call-parser"] == "qwen3_coder"
    assert spec.args["--enable-auto-tool-choice"] == "true"
    assert spec.args["--tensor-parallel-size"] == "2"
    assert "-tp" not in spec.args


def test_load_merge_respects_canonical_alias_collisions(lib):
    """`--tensor-parallel-size` in args and `-tp` in command are the same
    flag — args wins; no duplicate key in the merged view."""
    yaml_text = (
        "name: r2\n"
        "model: org/m\n"
        "args:\n"
        "  --tensor-parallel-size: '4'\n"
        "command: |\n"
        "  vllm serve org/m -tp 2\n"
    )
    lib.save_recipe_raw("r2", yaml_text)
    spec = lib.load_recipe("r2")
    assert spec.args["--tensor-parallel-size"] == "4"
    assert "-tp" not in spec.args


def test_load_recipe_text_prefers_override(lib):
    lib.save_recipe_raw("r1", "name: r1\nmodel: a/b\n")
    # Construct an override the same way LibraryService does.
    from sparkd import paths

    override_dir = paths.boxes_dir() / "box-x" / "overrides" / "recipes"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / "r1.yaml").write_text("name: r1\nmodel: a/b-override\n")
    assert "b-override" in lib.load_recipe_text("r1", box_id="box-x")
    assert "b-override" not in lib.load_recipe_text("r1")
