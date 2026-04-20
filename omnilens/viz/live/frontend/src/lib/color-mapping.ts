import {
  interpolateInferno,
  interpolateRdBu,
  interpolateViridis,
  interpolatePlasma,
} from "d3-scale-chromatic";

import type { MetricType, LayerMetrics } from "../types";

const SCALES: Record<MetricType, (t: number) => string> = {
  kl_divergence: interpolateInferno,
  loss_attribution: (t: number) => interpolateRdBu(1 - t), // flip so red = high
  activation: interpolateViridis,
  routing_importance: interpolatePlasma,
};

const LABELS: Record<MetricType, string> = {
  kl_divergence: "KL Divergence",
  loss_attribution: "Loss Attribution",
  activation: "Activation Magnitude",
  routing_importance: "Routing Importance",
};

export function getMetricLabel(metric: MetricType): string {
  return LABELS[metric];
}

export function getMetricValue(
  metrics: LayerMetrics | undefined,
  metric: MetricType
): number {
  if (!metrics) return 0;
  switch (metric) {
    case "activation":
      return Math.abs(metrics.activation.mean);
    case "kl_divergence":
      return metrics.kl_divergence;
    case "loss_attribution":
      return metrics.loss_attribution;
    case "routing_importance":
      return metrics.routing_importance;
  }
}

export function normalizeValue(
  value: number,
  min: number,
  max: number
): number {
  if (max <= min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

export function metricToColor(
  value: number,
  metric: MetricType,
  range: [number, number]
): string {
  const t = normalizeValue(value, range[0], range[1]);
  return SCALES[metric](t);
}

export function computeRange(
  values: number[]
): [number, number] {
  if (values.length === 0) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}

export function generateGradientStops(
  metric: MetricType,
  steps: number = 10
): { offset: string; color: string }[] {
  const stops: { offset: string; color: string }[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    stops.push({
      offset: `${(t * 100).toFixed(0)}%`,
      color: SCALES[metric](t),
    });
  }
  return stops;
}
