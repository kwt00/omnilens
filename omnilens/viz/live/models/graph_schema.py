from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str
    module_type: str
    parent_group: str | None = None
    num_params: int = 0
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    is_container: bool = False
    # Compression fields
    repeat_count: int = 1  # xN badge — how many identical blocks this represents
    compressed_ids: list[str] = []  # original layer IDs folded into this node
    tensor_volume: int = 0  # total number of elements flowing through (for edge width)


class GraphEdge(BaseModel):
    source: str
    target: str
    tensor_shape: list[int] | None = None
    tensor_volume: int = 0  # product of shape dims — drives edge width
    is_split: bool = False  # True if this edge fans out (data splits here)
    is_merge: bool = False  # True if this edge fans in (data merges here)


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    model_name: str
    total_params: int
    compressed: bool = False  # whether compression was applied
