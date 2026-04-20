from pydantic import BaseModel


class ActivationStats(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    histogram_counts: list[int] = []
    histogram_edges: list[float] = []


class LayerMetrics(BaseModel):
    layer_id: str
    activation: ActivationStats = ActivationStats()
    kl_divergence: float = 0.0
    loss_attribution: float = 0.0
    routing_importance: float = 0.0
    attention_entropy: float | None = None
    expert_utilization: list[float] | None = None
    skip_ratio: float | None = None


class MetricSnapshot(BaseModel):
    step: int = 0
    layers: dict[str, LayerMetrics] = {}


class FlowFrame(BaseModel):
    layer_id: str
    timestamp_ns: int
    shape: list[int]
    mean: float
    std: float
    min_val: float
    max_val: float


class FlowSequence(BaseModel):
    step: int
    frames: list[FlowFrame]
