export interface GraphNode {
  id: string;
  label: string;
  module_type: string;
  parent_group: string | null;
  num_params: number;
  input_shape: number[] | null;
  output_shape: number[] | null;
  is_container: boolean;
  repeat_count: number;
  compressed_ids: string[];
  tensor_volume: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  tensor_shape: number[] | null;
  tensor_volume: number;
  is_split: boolean;
  is_merge: boolean;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  model_name: string;
  total_params: number;
  compressed: boolean;
}

export interface ActivationStats {
  mean: number;
  std: number;
  min_val: number;
  max_val: number;
  histogram_counts: number[];
  histogram_edges: number[];
}

export interface LayerMetrics {
  layer_id: string;
  activation: ActivationStats;
  kl_divergence: number;
  loss_attribution: number;
  routing_importance: number;
  attention_entropy: number | null;
  expert_utilization: number[] | null;
  skip_ratio: number | null;
}

export interface MetricSnapshot {
  step: number;
  layers: Record<string, LayerMetrics>;
}

export interface FlowFrame {
  layer_id: string;
  timestamp_ns: number;
  shape: number[];
  mean: number;
  std: number;
  min_val: number;
  max_val: number;
}

export interface FlowSequence {
  step: number;
  frames: FlowFrame[];
}

export type MetricType =
  | "activation"
  | "kl_divergence"
  | "loss_attribution"
  | "routing_importance";

export interface WSMessage {
  type: "metrics" | "flow";
  data: MetricSnapshot | FlowSequence;
}
