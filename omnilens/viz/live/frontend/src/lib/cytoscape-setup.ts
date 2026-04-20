import cytoscape from "cytoscape";
// @ts-expect-error no types for cytoscape-dagre
import dagre from "cytoscape-dagre";

cytoscape.use(dagre);

export interface CytoscapeElements {
  nodes: cytoscape.ElementDefinition[];
  edges: cytoscape.ElementDefinition[];
}

import type { GraphResponse, GraphNode } from "../types";

const TYPE_LABELS: Record<string, string> = {
  Linear: "Linear",
  Conv2d: "Conv2d",
  Conv1d: "Conv1d",
  BatchNorm2d: "BatchNorm2d",
  BatchNorm1d: "BatchNorm1d",
  LayerNorm: "LayerNorm",
  Dropout: "Dropout",
  ReLU: "ReLU",
  GELU: "GELU",
  SiLU: "SiLU",
  Embedding: "Token embedding layer",
  MultiheadAttention: "Multi-head Attention",
  TransformerEncoderLayer: "Transformer Block",
  TransformerDecoderLayer: "Transformer Block",
  MaxPool2d: "MaxPool2d",
  AdaptiveAvgPool2d: "Adaptive AvgPool",
  Sequential: "Sequential",
  ModuleList: "ModuleList",
};


export function graphToElements(graph: GraphResponse): CytoscapeElements {
  const nodes: cytoscape.ElementDefinition[] = [];
  const edges: cytoscape.ElementDefinition[] = [];
  // Only include nodes that are referenced by edges (leaf nodes + IO)
  // Skip container nodes — they cause compound layout issues
  const edgeNodeIds = new Set<string>();
  for (const edge of graph.edges) {
    edgeNodeIds.add(edge.source);
    edgeNodeIds.add(edge.target);
  }

  for (const node of graph.nodes) {
    if (!edgeNodeIds.has(node.id)) continue; // skip orphan containers

    const isIO = node.module_type === "input" || node.module_type === "output";

    nodes.push({
      data: {
        id: node.id,
        label: formatLabel(node),
        moduleType: node.module_type,
        isContainer: false,
        repeatCount: node.repeat_count || 1,
        isIO,
      },
    });
  }

  for (const edge of graph.edges) {
    edges.push({
      data: {
        id: `${edge.source}->${edge.target}`,
        source: edge.source,
        target: edge.target,
      },
    });
  }

  return { nodes, edges };
}

function formatLabel(node: GraphNode): string {
  const cleanType =
    TYPE_LABELS[node.module_type] ||
    node.module_type
      .replace(/^F\./, "")
      .replace(/^method\./, "")
      .replace(/NonDynamicallyQuantizable/, "");

  return cleanType;
}

export const defaultStyle: cytoscape.StylesheetStyle[] = [
  {
    selector: "node",
    style: {
      label: "data(label)" as any,
      "text-valign": "center" as any,
      "text-halign": "center" as any,
      "text-wrap": "wrap" as any,
      "font-size": "11px",
      "font-family": "Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
      "font-weight": "500" as any,
      color: "#2a2a35",
      "background-color": "#ffffff",
      "background-opacity": 1,
      "border-width": 1,
      "border-color": "#d0d2da",
      shape: "roundrectangle" as any,
      width: 145,
      height: 36,
    },
  },
  // Repeated blocks — accent border
  {
    selector: "node[repeatCount > 1]",
    style: {
      "border-width": 1.5,
      "border-color": "#7c5cbf",
    },
  },
  // IO nodes
  {
    selector: "node[?isIO]",
    style: {
      width: 90,
      height: 30,
      "background-color": "#f5f6fa",
      "border-style": "dashed" as any,
      "border-color": "#b0b2c0",
      "font-size": "10px",
      color: "#8888a0",
      "font-weight": "normal" as any,
    },
  },
  // Edges
  {
    selector: "edge",
    style: {
      width: 1.2,
      "line-color": "#c8cad5",
      "target-arrow-color": "#c8cad5",
      "target-arrow-shape": "triangle" as any,
      "arrow-scale": 0.6,
      "curve-style": "bezier" as any,
    },
  },
  // Selected
  {
    selector: "node:selected",
    style: {
      "border-width": 2,
      "border-color": "#e06030",
    },
  },
];

export const layoutOptions = {
  name: "dagre",
  rankDir: "TB",
  nodeSep: 32,
  rankSep: 48,
  edgeSep: 16,
  animate: false,
  fit: true,
  padding: 50,
};
