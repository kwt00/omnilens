"""REST endpoints for the analysis panels — heatmap, attention, features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel
    from omnilens.methods.sae import SAE

router = APIRouter(prefix="/api")

_model: TappedModel | None = None
_sae: SAE | None = None
_sae_hook_point: str | None = None
_last_cache: dict | None = None
_last_tokens: list[str] | None = None


def set_model(model: TappedModel) -> None:
    global _model
    _model = model


def set_sae(sae: SAE, hook_point: str) -> None:
    global _sae, _sae_hook_point
    _sae = sae
    _sae_hook_point = hook_point


# --- Request/Response schemas ---


class RunRequest(BaseModel):
    text: str


class HeatmapCell(BaseModel):
    layer: int
    position: int
    token: str
    norm: float
    attn_norm: float | None = None
    mlp_norm: float | None = None


class HeatmapResponse(BaseModel):
    tokens: list[str]
    n_layers: int
    cells: list[HeatmapCell]
    top_predictions: list[str]  # model's top predictions for last token


class AttentionHead(BaseModel):
    head: int
    weights: list[list[float]]  # (seq, seq)


class AttentionResponse(BaseModel):
    layer: int
    tokens: list[str]
    heads: list[AttentionHead]


class FeatureActivation(BaseModel):
    feature_idx: int
    activation: float


class TokenFeatures(BaseModel):
    position: int
    token: str
    features: list[FeatureActivation]


class FeaturesResponse(BaseModel):
    hook_point: str
    tokens: list[str]
    token_features: list[TokenFeatures]


# --- Endpoints ---


@router.post("/run")
async def run_model(req: RunRequest) -> HeatmapResponse:
    """Run the model on input text and return activation heatmap data."""
    if _model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    global _last_cache, _last_tokens

    n_layers = _model._detect_n_layers()
    if n_layers is None:
        raise HTTPException(status_code=500, detail="Cannot detect number of layers")

    # Cache residual block outputs at every layer
    names = [f"layers.{i}.residual.block_out" for i in range(n_layers)]

    with torch.no_grad():
        logits, cache = _model.run_with_cache(text=req.text, names=names)

    # Get tokens
    input_ids = _model.tokenizer.encode(req.text)
    tokens = [_model.tokenizer.decode(t) for t in input_ids]
    _last_tokens = tokens
    _last_cache = {k: v.detach().cpu() for k, v in cache.items()}

    # Build heatmap cells
    cells = []
    for layer in range(n_layers):
        resid = cache[f"layers.{layer}.residual.block_out"][0]  # (seq, d_model)
        for pos in range(len(tokens)):
            cells.append(HeatmapCell(
                layer=layer,
                position=pos,
                token=tokens[pos],
                norm=resid[pos].float().norm().item(),
            ))

    # Top predictions for last token
    last_logits = logits[0, -1]
    top_ids = last_logits.topk(5).indices
    top_predictions = [_model.tokenizer.decode(t.item()) for t in top_ids]

    return HeatmapResponse(
        tokens=tokens,
        n_layers=n_layers,
        cells=cells,
        top_predictions=top_predictions,
    )


@router.get("/attention/{layer}")
async def get_attention(layer: int) -> AttentionResponse:
    """Get attention weights for a specific layer."""
    if _model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    if _last_tokens is None:
        raise HTTPException(status_code=400, detail="Run /api/run first")

    name = f"layers.{layer}.attention.weights"

    # Check if we already have it cached
    if _last_cache and name in _last_cache:
        weights = _last_cache[name]
    else:
        # Re-run to get attention weights
        with torch.no_grad():
            _, cache = _model.run_with_cache(
                text=" ".join(_last_tokens),  # reconstruct text
                names=[name],
            )
        weights = cache[name].detach().cpu()

    # weights shape: (1, heads, seq, seq)
    weights = weights[0].float()
    n_heads = weights.shape[0]

    heads = []
    for h in range(n_heads):
        heads.append(AttentionHead(
            head=h,
            weights=weights[h].tolist(),
        ))

    return AttentionResponse(
        layer=layer,
        tokens=_last_tokens,
        heads=heads,
    )


@router.post("/features")
async def get_features(req: RunRequest) -> FeaturesResponse:
    """Get SAE feature activations for the input text."""
    if _model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    if _sae is None or _sae_hook_point is None:
        raise HTTPException(status_code=404, detail="No SAE attached")

    with torch.no_grad():
        _, cache = _model.run_with_cache(
            text=req.text, names=[_sae_hook_point]
        )

    activations = cache[_sae_hook_point]
    features = _sae.encode(activations)  # (1, seq, n_features)
    features = features[0].detach().cpu()  # (seq, n_features)

    input_ids = _model.tokenizer.encode(req.text)
    tokens = [_model.tokenizer.decode(t) for t in input_ids]

    token_features = []
    for pos in range(len(tokens)):
        feat_vec = features[pos]
        active_indices = (feat_vec > 0).nonzero().squeeze(-1)
        active_list = []
        for idx in active_indices:
            idx = idx.item()
            active_list.append(FeatureActivation(
                feature_idx=idx,
                activation=feat_vec[idx].item(),
            ))
        # Sort by activation strength
        active_list.sort(key=lambda f: f.activation, reverse=True)
        # Keep top 20
        active_list = active_list[:20]

        token_features.append(TokenFeatures(
            position=pos,
            token=tokens[pos],
            features=active_list,
        ))

    return FeaturesResponse(
        hook_point=_sae_hook_point,
        tokens=tokens,
        token_features=token_features,
    )


@router.get("/info")
async def get_info():
    """Get model info and SAE status."""
    if _model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    info = {
        "model_name": _model._detect_architecture() or type(_model.model).__name__,
        "n_layers": _model._detect_n_layers(),
        "sae_attached": _sae is not None,
    }
    if _sae is not None:
        info["sae_hook_point"] = _sae_hook_point
        info["sae_n_features"] = _sae.n_features
    return info
