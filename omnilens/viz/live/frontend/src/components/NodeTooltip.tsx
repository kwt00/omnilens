import type { LayerMetrics, GraphNode } from "../types";

interface Props {
  nodeId: string;
  node: GraphNode | undefined;
  metrics: LayerMetrics | undefined;
  onClose: () => void;
}

export function NodeTooltip({ nodeId, node, metrics, onClose }: Props) {
  const shortId = nodeId.split(".").pop() || nodeId;

  return (
    <>
      <div className="tooltip-header">
        <h3>{shortId}</h3>
        <button className="tooltip-close" onClick={onClose}>x</button>
      </div>

      {node && (
        <div className="tooltip-section">
          <div className="tooltip-row">
            <span>Type</span>
            <span>{node.module_type}</span>
          </div>
          {node.repeat_count > 1 && (
            <div className="tooltip-row">
              <span>Repeated</span>
              <span className="repeat-badge">x{node.repeat_count}</span>
            </div>
          )}
          {node.num_params > 0 && (
            <div className="tooltip-row">
              <span>Params</span>
              <span>{node.num_params.toLocaleString()}</span>
            </div>
          )}
        </div>
      )}

      {metrics && (
        <div className="tooltip-section">
          <div className="tooltip-row">
            <span>Activity</span>
            <span>{Math.abs(metrics.activation.mean).toFixed(4)}</span>
          </div>
          <div className="tooltip-row">
            <span>KL Divergence</span>
            <span>{metrics.kl_divergence.toFixed(4)}</span>
          </div>
          {metrics.loss_attribution > 0.0001 && (
            <div className="tooltip-row">
              <span>Loss Impact</span>
              <span>{metrics.loss_attribution.toFixed(4)}</span>
            </div>
          )}
          {metrics.attention_entropy !== null && (
            <div className="tooltip-row">
              <span>Attn Entropy</span>
              <span>{metrics.attention_entropy.toFixed(3)}</span>
            </div>
          )}
          {metrics.skip_ratio !== null && (
            <div className="tooltip-row">
              <span>Skip Ratio</span>
              <span>{(metrics.skip_ratio * 100).toFixed(0)}%</span>
            </div>
          )}

          {metrics.activation.histogram_counts.length > 0 && (
            <div className="mini-histogram">
              <div className="histogram-label">Distribution</div>
              <div className="histogram-bars">
                {metrics.activation.histogram_counts.map((count, i) => {
                  const max = Math.max(...metrics.activation.histogram_counts);
                  const height = max > 0 ? (count / max) * 100 : 0;
                  return (
                    <div
                      key={i}
                      className="hist-bar"
                      style={{ height: `${height}%` }}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
