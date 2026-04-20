import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { GraphResponse, MetricSnapshot, MetricType } from "../types";
import {
  graphToElements,
  defaultStyle,
  layoutOptions,
} from "../lib/cytoscape-setup";
import {
  getMetricValue,
  metricToColor,
  computeRange,
} from "../lib/color-mapping";

interface Props {
  graph: GraphResponse;
  metrics: MetricSnapshot | null;
  activeMetric: MetricType;
  onSelectNode: (nodeId: string | null) => void;
  cyRef: React.MutableRefObject<cytoscape.Core | null>;
}

export function GraphCanvas({
  graph,
  metrics,
  activeMetric,
  onSelectNode,
  cyRef,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    const elements = graphToElements(graph);

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...elements.nodes, ...elements.edges],
      style: defaultStyle,
      layout: layoutOptions as any,
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
    });

    cy.on("tap", "node", (e) => {
      const nodeId = e.target.id();
      onSelectNode(nodeId);
    });

    cy.on("tap", (e) => {
      if (e.target === cy) {
        onSelectNode(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph]);

  // Update node colors when metrics or active metric changes
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !metrics) return;

    const values: number[] = [];
    const nodeMetrics = new Map<string, number>();

    cy.nodes().forEach((node) => {
      const nodeId = node.id();
      const layerMetrics = metrics.layers[nodeId];
      const value = getMetricValue(layerMetrics, activeMetric);
      values.push(value);
      nodeMetrics.set(nodeId, value);
    });

    const range = computeRange(values.filter((v) => v > 0));

    cy.batch(() => {
      cy.nodes().forEach((node) => {
        const nodeId = node.id();
        const value = nodeMetrics.get(nodeId) || 0;
        const isContainer = node.data("isContainer");

        if (!isContainer && value > 0) {
          const color = metricToColor(value, activeMetric, range);
          node.style("background-color", color);
        }
      });
    });
  }, [metrics, activeMetric, cyRef]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#f7f8fa",
      }}
    />
  );
}
