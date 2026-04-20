import { useRef, useEffect } from "react";
import type cytoscape from "cytoscape";
import type { MetricSnapshot } from "../types";

interface Props {
  cyRef: React.RefObject<cytoscape.Core | null>;
  metrics: MetricSnapshot | null;
}

/**
 * Canvas overlay that draws mini metric indicators directly
 * on each graph node — a small activity bar and value.
 */
export function MetricOverlay({ cyRef, metrics }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = parent.clientWidth * dpr;
      canvas.height = parent.clientHeight * dpr;
      canvas.style.width = `${parent.clientWidth}px`;
      canvas.style.height = `${parent.clientHeight}px`;
    };
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(canvas.parentElement!);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      const cy = cyRef.current;
      if (!canvas || !cy || !metrics) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

      // Find max values for normalization
      let maxAct = 0.001;
      let maxKL = 0.001;
      let maxLoss = 0.001;
      for (const m of Object.values(metrics.layers)) {
        maxAct = Math.max(maxAct, Math.abs(m.activation.mean));
        maxKL = Math.max(maxKL, m.kl_divergence);
        maxLoss = Math.max(maxLoss, m.loss_attribution);
      }

      cy.nodes().forEach((node) => {
        const id = node.id();
        const m = metrics.layers[id];
        if (!m) return;

        const pos = node.renderedPosition();
        const w = node.renderedWidth();
        const h = node.renderedHeight();

        // Draw a small bar underneath the node
        const barY = pos.y + h / 2 + 3;
        const barW = w * 0.8;
        const barH = 3;
        const barX = pos.x - barW / 2;

        // Activity bar (green)
        const actNorm = Math.min(1, Math.abs(m.activation.mean) / maxAct);
        ctx.fillStyle = "#d8dde4";
        ctx.beginPath();
        ctx.roundRect(barX, barY, barW, barH, 1.5);
        ctx.fill();

        if (actNorm > 0.01) {
          const color = actNorm > 0.7 ? "#e06030" : actNorm > 0.3 ? "#d4a030" : "#5a9a5a";
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.roundRect(barX, barY, barW * actNorm, barH, 1.5);
          ctx.fill();
        }

        // Small KL divergence dot (top-right of node) — bigger = more divergent
        if (m.kl_divergence > 0.01) {
          const klNorm = Math.min(1, m.kl_divergence / maxKL);
          const dotR = 2 + klNorm * 3;
          const dotX = pos.x + w / 2 - 4;
          const dotY = pos.y - h / 2 + 4;
          ctx.beginPath();
          ctx.arc(dotX, dotY, dotR, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(200, 80, 40, ${0.3 + klNorm * 0.5})`;
          ctx.fill();
        }

        // Loss attribution indicator (left side thin bar)
        if (m.loss_attribution > 0.0001) {
          const lossNorm = Math.min(1, m.loss_attribution / maxLoss);
          const lossH = h * 0.7 * lossNorm;
          const lossX = pos.x - w / 2 - 4;
          const lossY = pos.y + (h * 0.7) / 2 - lossH;
          ctx.fillStyle = `rgba(190, 60, 60, ${0.3 + lossNorm * 0.4})`;
          ctx.beginPath();
          ctx.roundRect(lossX, lossY, 2.5, lossH, 1);
          ctx.fill();
        }
      });
    };

    // Redraw on metrics change and on pan/zoom
    draw();

    const cy = cyRef.current;
    if (cy) {
      cy.on("pan zoom resize", draw);
      return () => { cy.off("pan zoom resize", draw); };
    }
  }, [metrics, cyRef]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        zIndex: 40,
      }}
    />
  );
}
