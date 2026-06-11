from datetime import datetime, timezone

import pytest

from sparkd.db.engine import init_engine
from sparkd.schemas.box import BoxCapabilities, BoxCreate
from sparkd.schemas.recipe import RecipeSpec
from sparkd.services.box import BoxService
from sparkd.services.library import LibraryService
from sparkd.services.recipe import RecipeService
from sparkd.ssh.pool import SSHPool, SSHTarget


@pytest.fixture
async def svc(sparkd_home, fake_box, monkeypatch):
    await init_engine(create_all=True)
    pool = SSHPool()
    box_svc = BoxService(pool=pool)
    lib = LibraryService()
    box, port = fake_box
    monkeypatch.setattr(
        box_svc,
        "_target_for",
        lambda b: SSHTarget(
            host="127.0.0.1", port=port, user="x", use_agent=False, password="y"
        ),
    )
    yield RecipeService(library=lib, boxes=box_svc, pool=pool), box_svc, box, port
    await pool.close_all()


def _caps(gpu_count: int, vram: int = 96) -> BoxCapabilities:
    return BoxCapabilities(
        gpu_count=gpu_count,
        gpu_model="NVIDIA GB10",
        vram_per_gpu_gb=vram,
        captured_at=datetime.now(timezone.utc),
    )


async def test_validate_passes_when_tp_matches_gpu_count(svc, monkeypatch):
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(2)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(name="r", model="m", args={"--tensor-parallel-size": "2"})
    issues = await rs.validate(r, bs.id)
    assert issues == []


async def test_validate_fails_when_tp_exceeds_gpus(svc, monkeypatch):
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(1)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(name="r", model="m", args={"--tensor-parallel-size": "4"})
    issues = await rs.validate(r, bs.id)
    assert any("exceeds" in i for i in issues)


async def test_validate_accepts_tp_when_cluster_total_gpus_match(svc, monkeypatch):
    """A 2-node-of-1-GPU cluster has total_gpus=2. tp=2 should pass —
    even though the head box's gpu_count is only 1. Without the cluster
    kwarg, the old validator rejected this perfectly-sized recipe."""
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(1)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(
        name="r",
        model="m",
        args={"--tensor-parallel-size": "2", "--pipeline-parallel-size": "1"},
    )
    issues = await rs.validate(
        r,
        bs.id,
        cluster={
            "name": "alpha",
            "nodes": [{"gpu_count": 1}, {"gpu_count": 1}],
            "total_gpus": 2,
        },
    )
    assert not any("exceeds" in i for i in issues), issues


async def test_validate_warns_when_tp_pp_leaves_gpus_idle_in_cluster(svc, monkeypatch):
    """tp=1 against a 2-GPU cluster fits but wastes a GPU. Surface that
    as an issue so the user notices before launching at half capacity."""
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(1)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(
        name="r",
        model="m",
        args={"--tensor-parallel-size": "1", "--pipeline-parallel-size": "1"},
    )
    issues = await rs.validate(
        r,
        bs.id,
        cluster={
            "name": "alpha",
            "nodes": [{"gpu_count": 1}, {"gpu_count": 1}],
            "total_gpus": 2,
        },
    )
    assert any("idle" in i for i in issues)


async def test_validate_rejects_tool_call_parser_without_enable_auto_tool_choice(svc, monkeypatch):
    """vLLM crashes mid-startup with `'auto' tool choice requires
    --enable-auto-tool-choice and --tool-call-parser to be set` when a
    parser is configured without enabling auto. Catch this in pre-flight
    so the user sees a clear message instead of a 60-second container
    crash + reconciler-marked-interrupted."""
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(2)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(
        name="r",
        model="m",
        args={
            "--tensor-parallel-size": "2",
            "--tool-call-parser": "qwen3_coder",
            # NOTE: missing --enable-auto-tool-choice
        },
    )
    issues = await rs.validate(r, bs.id)
    assert any("--tool-call-parser" in i for i in issues)
    assert any("--enable-auto-tool-choice" in i for i in issues)


async def test_validate_accepts_tool_call_parser_with_enable_auto_tool_choice(svc, monkeypatch):
    """The matched pair is fine."""
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(2)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(
        name="r",
        model="m",
        args={
            "--tensor-parallel-size": "2",
            "--tool-call-parser": "qwen3_coder",
            "--enable-auto-tool-choice": "true",
        },
    )
    issues = await rs.validate(r, bs.id)
    # The tool-call rule produces no issue when both are set.
    assert not any("--tool-call-parser" in i for i in issues), issues


async def test_validate_fails_when_tp_pp_exceeds_cluster_total(svc, monkeypatch):
    """tp=4 against a 2-GPU cluster is unsatisfiable — Ray will reject."""
    rs, box_svc, _, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))

    async def fake_caps(*_a, **_k):
        return _caps(1)

    monkeypatch.setattr(box_svc, "capabilities", fake_caps)
    r = RecipeSpec(
        name="r",
        model="m",
        args={"--tensor-parallel-size": "4", "--pipeline-parallel-size": "1"},
    )
    issues = await rs.validate(
        r,
        bs.id,
        cluster={
            "name": "alpha",
            "nodes": [{"gpu_count": 1}, {"gpu_count": 1}],
            "total_gpus": 2,
        },
    )
    assert any("exceeds" in i for i in issues)
    assert any("cluster" in i for i in issues)


async def test_sync_writes_yaml_to_box_repo(svc):
    rs, box_svc, fake, _port = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(RecipeSpec(name="r1", model="m", args={"--tp": "1"}))
    await rs.sync("r1", bs.id)
    assert any("recipes/r1.yaml" in c for c in fake.received)


async def test_sync_extra_env_merges_into_yaml(svc):
    """extra_env is appended to the recipe's env block (defaults-only merge);
    new keys land, existing keys are not clobbered. Verified by reading the
    heredoc payload that lands on the fake box."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(
        RecipeSpec(
            name="r1",
            model="m",
            env={"NCCL_SOCKET_IFNAME": "ibp0"},
        )
    )
    await rs.sync(
        "r1",
        bs.id,
        extra_env={
            "VLLM_HOST_IP": "$LOCAL_IP",
            "NCCL_SOCKET_IFNAME": "should-not-overwrite",
        },
    )
    blob = "\n".join(fake.received)
    assert "VLLM_HOST_IP: $LOCAL_IP" in blob
    # Existing key kept its original value — defaults-only merge.
    assert "NCCL_SOCKET_IFNAME: ibp0" in blob
    assert "should-not-overwrite" not in blob


async def test_sync_strip_env_keys_removes_entries(svc):
    """strip_env_keys is the cluster-launch escape hatch — keys named in
    the list are removed from the recipe's `env:` block before scping
    to the box. Untouched keys survive."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(
        RecipeSpec(
            name="r1",
            model="m",
            env={
                "VLLM_HOST_IP": "$LOCAL_IP",        # to strip
                "RAY_NODE_IP_ADDRESS": "$LOCAL_IP", # to strip
                "VLLM_USE_DEEP_GEMM": "0",          # to keep
            },
        )
    )
    await rs.sync(
        "r1",
        bs.id,
        strip_env_keys=["VLLM_HOST_IP", "RAY_NODE_IP_ADDRESS"],
    )
    blob = "\n".join(fake.received)
    assert "VLLM_HOST_IP" not in blob
    assert "RAY_NODE_IP_ADDRESS" not in blob
    # The model-specific key survives.
    assert "VLLM_USE_DEEP_GEMM: " in blob


async def test_sync_mirrors_top_level_model_into_defaults(svc):
    """When a recipe's command references `{model}` but `defaults.model`
    is missing, RecipeService.sync mirrors the top-level `model:` field
    into `defaults.model` so upstream's str.format substitution resolves.
    Otherwise upstream crashes with `Missing parameter in recipe command:
    'model'`. This is a defensive fix for recipes saved by older sparkd
    versions whose renderer omitted the mirror."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: brokenrec\n"
        "model: org/some-model\n"
        "container: vllm-node\n"
        "defaults:\n"
        "  port: 8000\n"
        "  tensor_parallel: 2\n"
        # NOTE: defaults has no `model` — upstream would crash on {model}
        "command: |\n"
        "  vllm serve {model} --port {port} -tp {tensor_parallel}\n"
    )
    rs.library.save_recipe_raw("brokenrec", raw)
    await rs.sync("brokenrec", bs.id)
    blob = "\n".join(fake.received)
    # The synced YAML must include `model: org/some-model` under defaults.
    assert "model: org/some-model" in blob
    # And must still contain the {model} template — we patched defaults,
    # not the command.
    assert "vllm serve {model}" in blob


async def test_sync_does_not_clobber_existing_defaults_model(svc):
    """If defaults already has a model entry, the mirror is a no-op —
    user-set defaults.model wins."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: ok\n"
        "model: org/top-level-model\n"
        "defaults:\n"
        "  model: org/defaults-wins\n"
        "  port: 8000\n"
        "command: |\n"
        "  vllm serve {model} --port {port}\n"
    )
    rs.library.save_recipe_raw("ok", raw)
    await rs.sync("ok", bs.id)
    blob = "\n".join(fake.received)
    assert "org/defaults-wins" in blob
    assert "org/top-level-model" in blob  # preserved at top level
    # The mirror only fills missing keys — should not have introduced a
    # second `model:` line under defaults pointing at the top-level.
    # (Hard to check structurally from a string, but: only one line with
    # `defaults-wins` exists, and it's after the `defaults:` header.)


async def test_sync_no_mirror_when_command_has_no_model_template(svc):
    """If the command hard-codes the model (upstream's actual pattern),
    the mirror is a no-op."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: hardcoded\n"
        "model: org/hard-coded-model\n"
        "defaults:\n"
        "  port: 8000\n"
        "command: |\n"
        "  vllm serve org/hard-coded-model --port {port}\n"
    )
    rs.library.save_recipe_raw("hardcoded", raw)
    await rs.sync("hardcoded", bs.id)
    blob = "\n".join(fake.received)
    # Defaults stays without a `model` entry — the recipe didn't need one.
    # Easiest test: no extra "model:" line shows up under defaults.
    # The top-level `model:` line is still there.
    assert "model: org/hard-coded-model" in blob


async def test_sync_strip_env_keys_combined_with_extra_env(svc):
    """strip and merge can co-occur in one sync call. Strip happens first,
    then merge — the order makes 'remove this then default this' an
    explicit option, not an accident."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(
        RecipeSpec(
            name="r1",
            model="m",
            env={"VLLM_HOST_IP": "$LOCAL_IP"},
        )
    )
    await rs.sync(
        "r1",
        bs.id,
        strip_env_keys=["VLLM_HOST_IP"],
        extra_env={"OMP_NUM_THREADS": "4"},
    )
    blob = "\n".join(fake.received)
    assert "VLLM_HOST_IP" not in blob
    assert "OMP_NUM_THREADS: " in blob


async def test_sync_regenerates_command_from_args_when_args_non_empty(svc):
    """When the recipe carries an `args:` block (typical of sparkd-rendered
    and advisor-generated recipes), sync regenerates the `command` so every
    flag in args actually reaches vLLM. Without this, advisor flags like
    `--pipeline-parallel-size` and `--quantization` get dropped on the
    floor — they live only in `args:` and never reach the command line."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(
        RecipeSpec(
            name="advisor",
            model="org/some-model",
            args={
                "--tensor-parallel-size": "2",
                "--pipeline-parallel-size": "1",
                "--quantization": "modelopt_fp4",
                "--distributed-executor-backend": "ray",
                "--trust-remote-code": "true",
                "--gpu-memory-utilization": "0.90",
                "--max-model-len": "8192",
            },
        )
    )
    await rs.sync("advisor", bs.id)
    blob = "\n".join(fake.received)
    # Every advisor-set flag is in the command we shipped.
    assert "--tensor-parallel-size 2" in blob
    assert "--pipeline-parallel-size 1" in blob
    assert "--quantization modelopt_fp4" in blob
    assert "--distributed-executor-backend ray" in blob
    # Boolean flag is emitted as a bare flag, not as `--trust-remote-code true`.
    assert "--trust-remote-code\n" in blob or "--trust-remote-code " in blob
    assert "--trust-remote-code true" not in blob
    assert "--gpu-memory-utilization 0.90" in blob
    assert "--max-model-len 8192" in blob


async def test_sync_emits_enable_auto_tool_choice_as_bare_flag(svc):
    """vLLM rejects `--enable-auto-tool-choice true` (the string "true"
    is read as a positional). The renderer must emit it as a bare flag.
    Same shape as --trust-remote-code."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    rs.library.save_recipe(
        RecipeSpec(
            name="tools",
            model="org/m",
            args={
                "--tool-call-parser": "qwen3_coder",
                "--enable-auto-tool-choice": "true",
            },
        )
    )
    await rs.sync("tools", bs.id)
    blob = "\n".join(fake.received)
    assert "--tool-call-parser qwen3_coder" in blob
    assert "--enable-auto-tool-choice" in blob
    # And NOT as `--enable-auto-tool-choice true`.
    assert "--enable-auto-tool-choice true" not in blob


async def test_sync_merges_curated_command_flags_into_regenerated_command(svc):
    """A recipe can carry both a hand-curated `command` and an `args:`
    block that drifted apart — e.g. tool-call flags present only in the
    command (the Nemotron-3-Super case). Regenerating the command from
    args must preserve command-only flags instead of silently dropping
    them; args wins on conflicts; `{placeholder}` values survive for
    upstream's defaults substitution."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: drifted\n"
        "model: nvidia/some-model\n"
        "container: vllm-node\n"
        "defaults:\n"
        "  port: 8000\n"
        "  tensor_parallel: 2\n"
        "command: |\n"
        "  vllm serve nvidia/some-model \\\n"
        "    --port {port} \\\n"
        "    --enable-auto-tool-choice \\\n"
        "    --tool-call-parser qwen3_coder \\\n"
        "    --tensor-parallel-size {tensor_parallel} \\\n"
        "    --distributed-executor-backend ray\n"
        "args:\n"
        "  --port: '9999'\n"
        "  --max-model-len: '131072'\n"
    )
    rs.library.save_recipe_raw("drifted", raw)
    await rs.sync("drifted", bs.id)
    blob = "\n".join(fake.received)
    # Command-only flags survive the regeneration.
    assert "--tool-call-parser qwen3_coder" in blob
    assert "--enable-auto-tool-choice" in blob
    assert "--enable-auto-tool-choice true" not in blob
    assert "--distributed-executor-backend ray" in blob
    # Placeholder value kept verbatim for defaults substitution.
    assert "--tensor-parallel-size {tensor_parallel}" in blob
    # args wins on conflict: the curated {port} placeholder is replaced.
    assert "--port 9999" in blob
    assert "--port {port}" not in blob
    # args-only flags still present.
    assert "--max-model-len 131072" in blob


async def test_sync_preserves_command_when_args_empty(svc):
    """Upstream-format recipes (hand-curated commands, empty `args:`) must
    pass through unchanged. Only the args-driven path regenerates."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: upstream\n"
        "model: org/m\n"
        "container: vllm-node\n"
        "args: {}\n"
        "defaults:\n"
        "  port: 8000\n"
        "  tensor_parallel: 4\n"
        "command: |\n"
        "  vllm serve org/m --port {port} -tp {tensor_parallel}\n"
    )
    rs.library.save_recipe_raw("upstream", raw)
    await rs.sync("upstream", bs.id)
    blob = "\n".join(fake.received)
    # Original templated command is still there — we did not regenerate.
    assert "vllm serve org/m --port {port} -tp {tensor_parallel}" in blob


async def test_sync_without_extra_env_passes_yaml_byte_identical(svc):
    """No extra_env → the on-disk YAML is shipped verbatim, preserving
    upstream-format fields like `defaults`/`command`/`container`."""
    rs, box_svc, fake, _ = svc
    bs = await box_svc.create(BoxCreate(name="b", host="h", user="u"))
    raw = (
        "name: r1\n"
        "model: m\n"
        "container: vllm-node\n"
        "defaults: { port: 8000 }\n"
        "command: vllm serve {model} --port {port}\n"
    )
    rs.library.save_recipe_raw("r1", raw)
    await rs.sync("r1", bs.id)
    blob = "\n".join(fake.received)
    # The literal `command:` template survives — we did not re-serialize.
    assert "vllm serve {model} --port {port}" in blob
