from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from sparkd.advisor.providers import DEFAULT_ANTHROPIC_MODEL
from sparkd.advisor.providers import PROVIDERS as PROVIDERS_CATALOG
from sparkd.errors import NotFoundError, ValidationError
from sparkd.hardware import default_dgx_spark_caps
from sparkd.schemas.advisor import AdvisorSession
from sparkd.schemas.box import BoxCapabilities
from sparkd.services import advisor_config
from sparkd.services.advisor import AdvisorService
from sparkd.services.box import BoxService
from sparkd.services.hf_catalog import HFCatalogService
from sparkd.services.library import LibraryService
from sparkd.services.targets import CLUSTER_PREFIX, resolve_target

router = APIRouter(prefix="/advisor", tags=["advisor"])


def _svc(request: Request) -> AdvisorService:
    return request.app.state.advisor


def _hf(request: Request) -> HFCatalogService:
    return request.app.state.hf


def _boxes(request: Request) -> BoxService:
    return request.app.state.boxes


def _lib(request: Request) -> LibraryService:
    return request.app.state.library


async def _resolve_caps(
    target_box_id: str | None, boxes: BoxService
) -> BoxCapabilities:
    """Capabilities of the head box, falling back to canonical defaults."""
    if not target_box_id:
        return default_dgx_spark_caps()
    try:
        resolved = await resolve_target(target_box_id, boxes)
        return await boxes.capabilities(resolved.head_box.id)
    except Exception:  # noqa: BLE001  — degrade gracefully
        return default_dgx_spark_caps()


async def _resolve_cluster(
    target_box_id: str | None, boxes: BoxService
) -> dict | None:
    """Topology dict for cluster targets; None for single-box / unset."""
    if not target_box_id or not target_box_id.startswith(CLUSTER_PREFIX):
        return None
    try:
        resolved = await resolve_target(target_box_id, boxes)
    except Exception:  # noqa: BLE001
        return None
    if resolved.kind != "cluster":
        return None
    nodes: list[dict] = []
    for box in resolved.members:
        try:
            caps = await boxes.capabilities(box.id)
        except Exception:  # noqa: BLE001
            caps = default_dgx_spark_caps()
        nodes.append(
            {
                "name": box.name,
                "host": box.host,
                "gpu_count": caps.gpu_count,
                "gpu_model": caps.gpu_model,
                "vram_gb": caps.vram_per_gpu_gb,
                "ib": caps.ib_interface,
            }
        )
    total_gpus = sum(n["gpu_count"] for n in nodes)
    total_vram = sum(n["gpu_count"] * n["vram_gb"] for n in nodes)
    return {
        "name": resolved.cluster_name,
        "nodes": nodes,
        "total_gpus": total_gpus,
        "total_vram_gb": total_vram,
    }


class CreateSessionBody(BaseModel):
    kind: str
    target_box_id: str | None = None
    target_recipe_name: str | None = None
    hf_model_id: str | None = None


class CreateSessionResponse(BaseModel):
    id: str


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionBody, svc: AdvisorService = Depends(_svc)
) -> CreateSessionResponse:
    if body.kind not in {"recipe", "optimize", "mod"}:
        raise ValidationError(f"invalid kind: {body.kind!r}")
    sid = await svc.create_session(
        kind=body.kind,
        target_box_id=body.target_box_id,
        target_recipe_name=body.target_recipe_name,
        hf_model_id=body.hf_model_id,
    )
    return CreateSessionResponse(id=sid)


@router.get("/sessions/{session_id}", response_model=AdvisorSession)
async def get_session(
    session_id: str, svc: AdvisorService = Depends(_svc)
) -> AdvisorSession:
    return await svc.get_session(session_id)


class GenerateRecipeBody(BaseModel):
    user_msg: str | None = None


@router.post("/sessions/{session_id}/recipe")
async def generate_recipe(
    session_id: str,
    body: GenerateRecipeBody,
    svc: AdvisorService = Depends(_svc),
    hf: HFCatalogService = Depends(_hf),
    boxes: BoxService = Depends(_boxes),
) -> dict:
    sess = await svc.get_session(session_id)
    if not sess.hf_model_id:
        raise ValidationError("session has no hf_model_id")
    info = await hf.fetch(sess.hf_model_id)
    caps = await _resolve_caps(sess.target_box_id, boxes)
    cluster = await _resolve_cluster(sess.target_box_id, boxes)
    draft = None
    deltas: list[str] = []
    async for ev in svc.generate_recipe(
        session_id,
        info=info,
        caps=caps,
        user_msg=body.user_msg,
        cluster=cluster,
    ):
        if ev["type"] == "delta":
            deltas.append(ev["text"])
        elif ev["type"] == "draft":
            draft = ev["draft"]
        elif ev["type"] == "error":
            raise ValidationError(ev["message"])
    return {"draft": draft, "text": "".join(deltas)}


class OptimizeBody(BaseModel):
    goals: list[str] = Field(default_factory=list)
    user_msg: str | None = None


@router.post("/sessions/{session_id}/optimize")
async def optimize_recipe(
    session_id: str,
    body: OptimizeBody,
    svc: AdvisorService = Depends(_svc),
    boxes: BoxService = Depends(_boxes),
    lib: LibraryService = Depends(_lib),
) -> dict:
    sess = await svc.get_session(session_id)
    if not sess.target_recipe_name:
        raise ValidationError("session needs target_recipe_name")
    recipe = lib.load_recipe(sess.target_recipe_name, box_id=sess.target_box_id)
    caps = await _resolve_caps(sess.target_box_id, boxes)
    cluster = await _resolve_cluster(sess.target_box_id, boxes)
    draft = None
    deltas: list[str] = []
    async for ev in svc.optimize_recipe(
        session_id,
        recipe=recipe,
        caps=caps,
        goals=body.goals,
        user_msg=body.user_msg,
        cluster=cluster,
    ):
        if ev["type"] == "delta":
            deltas.append(ev["text"])
        elif ev["type"] == "draft":
            draft = ev["draft"]
        elif ev["type"] == "error":
            raise ValidationError(ev["message"])
    return {"draft": draft, "text": "".join(deltas)}


class ProposeModBody(BaseModel):
    error_log: str
    user_msg: str | None = None


@router.post("/sessions/{session_id}/mod")
async def propose_mod(
    session_id: str,
    body: ProposeModBody,
    svc: AdvisorService = Depends(_svc),
) -> dict:
    sess = await svc.get_session(session_id)
    if not sess.hf_model_id:
        raise ValidationError("session has no hf_model_id")
    draft = None
    deltas: list[str] = []
    async for ev in svc.propose_mod(
        session_id,
        error_log=body.error_log,
        model_id=sess.hf_model_id,
        user_msg=body.user_msg,
    ):
        if ev["type"] == "delta":
            deltas.append(ev["text"])
        elif ev["type"] == "draft":
            draft = ev["draft"]
        elif ev["type"] == "error":
            raise ValidationError(ev["message"])
    return {"draft": draft, "text": "".join(deltas)}


class SetupBody(BaseModel):
    """Legacy setup — kept for backward compat. Equivalent to switching the
    active provider to anthropic and saving the key."""

    anthropic_api_key: str


@router.post("/setup")
def setup(body: SetupBody, request: Request) -> dict:
    if not body.anthropic_api_key.strip():
        raise ValidationError("anthropic_api_key is required")
    key = body.anthropic_api_key.strip()
    advisor_config.set_api_key("anthropic", key)
    cfg = advisor_config.load_config()
    cfg.active_provider = "anthropic"
    if not cfg.get_state("anthropic").model:
        cfg.get_state("anthropic").model = DEFAULT_ANTHROPIC_MODEL
    advisor_config.save_config(cfg)
    request.app.state.advisor.port = advisor_config.build_port(cfg)
    return {"ok": True}


@router.get("/status")
def status(request: Request) -> dict:
    cfg = advisor_config.load_config()
    pdef_id = cfg.active_provider
    state = cfg.get_state(pdef_id)
    port = getattr(request.app.state.advisor, "port", None)
    return {
        "configured": port is not None,
        "active_provider": pdef_id,
        "active_model": state.model,
    }


@router.get("/providers")
def list_providers() -> dict:
    cfg = advisor_config.load_config()
    return {
        "active_provider": cfg.active_provider,
        "providers": advisor_config.provider_summary(),
        "configured": [
            pid
            for pid, _ in cfg.providers.items()
            if advisor_config.has_api_key(pid)
        ],
    }


class ProviderConfigBody(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None  # optional — only sent when changing it
    set_active: bool = True


@router.put("/config")
def put_config(body: ProviderConfigBody, request: Request) -> dict:
    pdef = next((p for p in PROVIDERS_CATALOG if p.id == body.provider), None)
    if pdef is None:
        raise NotFoundError("provider", body.provider)
    if not body.model.strip():
        raise ValidationError("model is required")
    if pdef.requires_key:
        existing_key = advisor_config.get_api_key(body.provider)
        if not body.api_key and not existing_key:
            raise ValidationError(
                f"{body.provider}: api_key required (provider does not allow anonymous access)"
            )
    if body.api_key:
        advisor_config.set_api_key(body.provider, body.api_key.strip())
    cfg = advisor_config.load_config()
    state = cfg.get_state(body.provider)
    state.model = body.model.strip()
    state.base_url = (body.base_url or "").strip() or None
    if body.set_active:
        cfg.active_provider = body.provider
    advisor_config.save_config(cfg)
    request.app.state.advisor.port = advisor_config.build_port(cfg)
    return {
        "ok": True,
        "active_provider": cfg.active_provider,
        "active_model": cfg.get_state(cfg.active_provider).model,
    }


@router.get("/config")
def get_config() -> dict:
    cfg = advisor_config.load_config()
    return {
        "active_provider": cfg.active_provider,
        "providers": {
            pid: {"model": s.model, "base_url": s.base_url}
            for pid, s in cfg.providers.items()
        },
    }
