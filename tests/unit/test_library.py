import pytest

from sparkd.errors import NotFoundError, ValidationError
from sparkd.schemas.recipe import RecipeSpec
from sparkd.services.library import LibraryService


@pytest.fixture
def lib(sparkd_home):
    return LibraryService()


def test_save_then_load_recipe(lib):
    """Saved args round-trip. The loaded view also surfaces the flags from
    the rendered command template (host/port/gmu/max-model-len/-tp from
    `defaults`) — those reach vLLM at launch, so the parsed view shows
    them rather than pretending the recipe only has `--foo`."""
    r = RecipeSpec(name="r1", model="m", args={"--foo": "bar"})
    lib.save_recipe(r)
    got = lib.load_recipe("r1")
    assert got.args["--foo"] == "bar"
    # Template-derived flags surface in the effective view, canonicalized.
    assert got.args["--host"] == "0.0.0.0"
    assert got.args["--port"] == "8000"
    assert got.args["--tensor-parallel-size"] == "1"


def test_load_missing_raises(lib):
    with pytest.raises(NotFoundError):
        lib.load_recipe("nope")


def test_list_recipes_returns_canonical(lib):
    lib.save_recipe(RecipeSpec(name="a", model="m"))
    lib.save_recipe(RecipeSpec(name="b", model="m"))
    names = [r.name for r in lib.list_recipes()]
    assert sorted(names) == ["a", "b"]


def test_effective_view_merges_overrides(lib):
    lib.save_recipe(RecipeSpec(name="r1", model="m", args={"--tp": "1"}))
    lib.save_recipe_override(
        "box-x", RecipeSpec(name="r1", model="m", args={"--tp": "2"})
    )
    eff = lib.load_recipe("r1", box_id="box-x")
    assert eff.args["--tp"] == "2"


def test_save_recipe_rejects_path_traversal(lib):
    with pytest.raises(ValidationError):
        lib.save_recipe(RecipeSpec(name="../evil", model="m"))


def test_delete_recipe(lib):
    lib.save_recipe(RecipeSpec(name="r1", model="m"))
    lib.delete_recipe("r1")
    with pytest.raises(NotFoundError):
        lib.load_recipe("r1")


def test_save_recipe_includes_model_in_defaults_block(lib):
    """Sparkd's renderer emits `defaults.model = spec.model` so that the
    `command` template's `{model}` substitution resolves under upstream's
    run-recipe.py. Without it, every cluster launch crashes with
    `Missing parameter in recipe command: 'model'` because upstream's
    str.format reads from `defaults`, not the top-level `model:` field."""
    from sparkd import paths

    lib.save_recipe(RecipeSpec(name="r1", model="org/the-model"))
    yaml_text = (paths.library() / "recipes" / "r1.yaml").read_text()
    assert "model: org/the-model" in yaml_text  # top-level
    # defaults block must contain a `model:` line.
    assert "defaults:" in yaml_text
    defaults_section = yaml_text.split("defaults:", 1)[1].split("\n\n", 1)[0]
    assert "model: org/the-model" in defaults_section
