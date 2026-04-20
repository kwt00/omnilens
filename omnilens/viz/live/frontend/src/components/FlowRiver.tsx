import {
  forwardRef,
  useImperativeHandle,
  useRef,
  useEffect,
  useCallback,
} from "react";
import type cytoscape from "cytoscape";
import type { FlowSequence, GraphResponse } from "../types";

export interface FlowRiverHandle {
  play: (flow: FlowSequence, cy: cytoscape.Core, speed: number) => void;
  stop: () => void;
}

interface FlowRiverProps {
  graph: GraphResponse;
}

type DataShape = "tokens" | "vector" | "matrix" | "sparse" | "normalized" | "logits";

const SHAPE_LABEL: Record<DataShape, string> = {
  tokens: "tokens",
  vector: "tensor",
  matrix: "attn weights",
  sparse: "masked",
  normalized: "normalized",
  logits: "logits",
};

const SHAPE_COLOR: Record<DataShape, string> = {
  tokens:     "#4a90c4",
  vector:     "#5a9a5a",
  matrix:     "#c08040",
  sparse:     "#888888",
  normalized: "#8060b0",
  logits:     "#c05050",
};

function getDataShape(nodeId: string, moduleType: string): DataShape {
  const t = (moduleType || "").toLowerCase();
  const id = (nodeId || "").toLowerCase();
  if (t === "input" || t.includes("embed")) return "tokens";
  if (t.includes("attention") || id.includes("attn")) return "matrix";
  if (t.includes("dropout")) return "sparse";
  if (t.includes("norm")) return "normalized";
  if (t === "output") return "logits";
  return "vector";
}

interface Segment {
  srcId: string;
  tgtId: string;
  shape: DataShape;
  /** Which loop iteration this segment is in (0-based), -1 if not in a loop */
  loopIdx: number;
  /** Total loop count for display */
  totalLoops: number;
}

/**
 * One data object at a time travels the full path, morphing shape
 * at each layer. Clean, readable, not cluttered.
 */
export const FlowRiver = forwardRef<FlowRiverHandle, FlowRiverProps>((props, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const playing = useRef(false);
  const cyRef = useRef<cytoscape.Core | null>(null);

  // Current animation state
  const path = useRef<Segment[]>([]);
  const segIdx = useRef(0);
  const segT = useRef(0);
  const speedRef = useRef(1);
  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  const clear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const stop = useCallback(() => {
    playing.current = false;
    cancelAnimationFrame(rafRef.current);
    if (restartTimer.current) {
      clearTimeout(restartTimer.current);
      restartTimer.current = null;
    }
    clear();
  }, [clear]);

  function getPos(cy: cytoscape.Core, id: string) {
    const node = cy.getElementById(id);
    if (node.length === 0) return null;
    return node.renderedPosition();
  }

  function drawShape(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    shape: DataShape,
    alpha: number
  ) {
    const color = SHAPE_COLOR[shape];
    ctx.globalAlpha = alpha;

    const S = 14; // base size unit

    switch (shape) {
      case "tokens": {
        // Three small cards with text lines
        for (let i = 0; i < 3; i++) {
          const bx = x - 18 + i * 14;
          const by = y - S * 0.4;
          ctx.fillStyle = "#f0f4f8";
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.roundRect(bx, by, 10, S * 0.8, 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.globalAlpha = alpha * 0.6;
          ctx.fillRect(bx + 2, by + 2.5, 6, 1.2);
          ctx.fillRect(bx + 2, by + 5.5, 4, 1.2);
          ctx.globalAlpha = alpha;
        }
        break;
      }

      case "vector": {
        // Single vertical bar with gradient segments
        const w = 8, h = S * 2;
        const sx = x - w / 2, sy = y - h / 2;
        ctx.fillStyle = "#eaf4ea";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(sx, sy, w, h, 2);
        ctx.fill();
        ctx.stroke();
        for (let i = 0; i < 8; i++) {
          const val = 0.2 + Math.abs(Math.sin(i * 1.5)) * 0.7;
          ctx.fillStyle = `rgba(70, 150, 70, ${val * alpha})`;
          ctx.fillRect(sx + 1, sy + 1 + i * (h / 8), w - 2, h / 8 - 1);
        }
        break;
      }

      case "matrix": {
        // 4x4 heatmap grid
        const gs = S * 1.4;
        const cs = gs / 4;
        const sx = x - gs / 2, sy = y - gs / 2;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        for (let r = 0; r < 4; r++) {
          for (let c = 0; c < 4; c++) {
            const v = 0.15 + Math.abs(Math.sin(r * 2.3 + c * 1.7)) * 0.75;
            ctx.fillStyle = `rgba(190, 120, 50, ${v * alpha})`;
            ctx.fillRect(sx + c * cs, sy + r * cs, cs - 0.8, cs - 0.8);
          }
        }
        ctx.strokeRect(sx, sy, gs, gs);
        break;
      }

      case "sparse": {
        // Vector with Xs for dropped elements
        const w = 8, h = S * 2;
        const sx = x - w / 2, sy = y - h / 2;
        ctx.fillStyle = "#f4f4f4";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(sx, sy, w, h, 2);
        ctx.fill();
        ctx.stroke();
        for (let i = 0; i < 8; i++) {
          const dropped = i % 3 === 1;
          if (dropped) {
            ctx.strokeStyle = "#cc4444";
            ctx.lineWidth = 0.8;
            const cy = sy + 1 + i * (h / 8) + h / 16;
            ctx.beginPath();
            ctx.moveTo(sx + 1.5, cy - 1.5);
            ctx.lineTo(sx + w - 1.5, cy + 1.5);
            ctx.moveTo(sx + w - 1.5, cy - 1.5);
            ctx.lineTo(sx + 1.5, cy + 1.5);
            ctx.stroke();
            ctx.strokeStyle = color;
          } else {
            ctx.fillStyle = `rgba(100, 100, 110, ${0.5 * alpha})`;
            ctx.fillRect(sx + 1, sy + 1 + i * (h / 8), w - 2, h / 8 - 1);
          }
        }
        break;
      }

      case "normalized": {
        // Even bars — all same height
        const w = 8, h = S * 2;
        const sx = x - w / 2, sy = y - h / 2;
        ctx.fillStyle = "#f0eaf6";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(sx, sy, w, h, 2);
        ctx.fill();
        ctx.stroke();
        for (let i = 0; i < 8; i++) {
          ctx.fillStyle = `rgba(120, 90, 170, ${0.5 * alpha})`;
          ctx.fillRect(sx + 1, sy + 1 + i * (h / 8), w - 2, h / 8 - 1);
        }
        break;
      }

      case "logits": {
        // Horizontal bars of different lengths
        const bw = S * 2, bh = S * 1.2;
        const sx = x - bw / 2, sy = y - bh / 2;
        ctx.fillStyle = "#f8f0f0";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(sx, sy, bw, bh, 2);
        ctx.fill();
        ctx.stroke();
        const probs = [0.55, 0.22, 0.13, 0.1];
        for (let i = 0; i < 4; i++) {
          ctx.fillStyle = `rgba(190, 60, 60, ${(0.3 + probs[i]) * alpha})`;
          ctx.fillRect(sx + 1.5, sy + 1 + i * (bh / 4), probs[i] * (bw - 3), bh / 4 - 1.5);
        }
        break;
      }
    }

    // Label below
    ctx.globalAlpha = alpha * 0.7;
    ctx.fillStyle = "#444";
    ctx.font = "9px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(SHAPE_LABEL[shape], x, y + S + 6);

    ctx.globalAlpha = 1;
    ctx.textAlign = "start";
  }

  function startPath() {
    segIdx.current = 0;
    segT.current = 0;
  }

  const render = useCallback(() => {
    if (!playing.current) return;

    const canvas = canvasRef.current;
    const cy = cyRef.current;
    if (!canvas || !cy) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

    const segs = path.current;
    if (segs.length === 0) return;

    const idx = segIdx.current;
    if (idx >= segs.length) {
      // Done — restart after a pause
      clear();
      restartTimer.current = setTimeout(() => {
        if (!playing.current) return;
        startPath();
        rafRef.current = requestAnimationFrame(render);
      }, 1500 / speedRef.current);
      return;
    }

    const seg = segs[idx];
    const sp = getPos(cy, seg.srcId);
    const tp = getPos(cy, seg.tgtId);

    if (!sp || !tp) {
      // Skip bad segment
      segIdx.current++;
      rafRef.current = requestAnimationFrame(render);
      return;
    }

    // Advance
    segT.current += (0.008 + 0.002 * Math.random()) * speedRef.current;
    const t = segT.current;

    if (t >= 1) {
      // Move to next segment
      segIdx.current++;
      segT.current = 0;
      rafRef.current = requestAnimationFrame(render);
      return;
    }

    // Position on bezier
    const dx = tp.x - sp.x;
    const dy = tp.y - sp.y;
    const cx = (sp.x + tp.x) / 2 - dy * 0.04;
    const cy2 = (sp.y + tp.y) / 2 + dx * 0.04;
    const u = 1 - t;
    const px = u * u * sp.x + 2 * u * t * cx + t * t * tp.x;
    const py = u * u * sp.y + 2 * u * t * cy2 + t * t * tp.y;

    // Fade
    let alpha = 0.95;
    if (t < 0.1) alpha *= t / 0.1;
    if (t > 0.9) alpha *= (1 - t) / 0.1;

    // Draw a faint trail on the edge
    ctx.beginPath();
    ctx.moveTo(sp.x, sp.y);
    ctx.quadraticCurveTo(cx, cy2, tp.x, tp.y);
    ctx.strokeStyle = `rgba(${hexToRgb(SHAPE_COLOR[seg.shape])}, ${alpha * 0.12})`;
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.stroke();

    // Draw the data shape
    drawShape(ctx, px, py, seg.shape, alpha);

    // Show loop counter when in a cycle
    if (seg.loopIdx > 0) {
      ctx.globalAlpha = alpha * 0.8;
      ctx.fillStyle = "#7c5cbf";
      ctx.font = "bold 10px Inter, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        `loop ${seg.loopIdx}/${seg.totalLoops}`,
        px,
        py - 20
      );
      ctx.globalAlpha = 1;
      ctx.textAlign = "start";
    }

    rafRef.current = requestAnimationFrame(render);
  }, [clear]);

  const play = useCallback(
    (_flow: FlowSequence, cy: cytoscape.Core, speed: number) => {
      if (playing.current) return;

      stop();
      playing.current = true;
      cyRef.current = cy;
      speedRef.current = speed;

      // Build adjacency
      const adj = new Map<string, string[]>();
      cy.edges().forEach((e) => {
        const src = e.source().id();
        const tgt = e.target().id();
        const list = adj.get(src) || [];
        list.push(tgt);
        adj.set(src, list);
      });

      const types = new Map<string, string>();
      cy.nodes().forEach((n) => {
        types.set(n.id(), n.data("moduleType") || "");
      });

      // Find root
      const hasIncoming = new Set<string>();
      cy.edges().forEach((e) => { hasIncoming.add(e.target().id()); });
      let root = "";
      for (const [nodeId] of adj) {
        if (!hasIncoming.has(nodeId)) { root = nodeId; break; }
      }
      if (!root) return;

      // Build path: follow graph, loop through cycles 2x before exiting
      // The repeat count from the graph tells us the real count (e.g. x12)
      const CYCLE_LOOPS = 2;
      const MAX_STEPS = 60;
      const segments: Segment[] = [];
      let current = root;
      const visitCount = new Map<string, number>();

      // Find the actual repeat count from graph nodes
      let repeatCount = 12;
      for (const n of props.graph.nodes) {
        if (n.repeat_count > 1) { repeatCount = n.repeat_count; break; }
      }

      let currentLoop = 0; // which loop iteration we're on

      for (let i = 0; i < MAX_STEPS; i++) {
        visitCount.set(current, (visitCount.get(current) || 0) + 1);
        const nexts = adj.get(current) || [];
        if (nexts.length === 0) break;

        let next: string | null = null;
        let isLoopBack = false;

        if (nexts.length === 1) {
          const c = visitCount.get(nexts[0]) || 0;
          if (c < CYCLE_LOOPS) next = nexts[0];
        } else {
          const loopBacks = nexts.filter((n) => visitCount.has(n) && visitCount.get(n)! > 0);
          const exits = nexts.filter((n) => !visitCount.has(n) || visitCount.get(n)! === 0);

          if (loopBacks.length > 0) {
            const lb = loopBacks[0];
            const c = visitCount.get(lb) || 0;
            if (c < CYCLE_LOOPS) {
              next = lb;
              isLoopBack = true;
            } else if (exits.length > 0) {
              next = exits[0];
            }
          } else if (exits.length > 0) {
            next = exits[0];
          }
        }

        if (!next) break;

        if (isLoopBack) currentLoop++;

        const shape = getDataShape(next, types.get(next) || "");
        segments.push({
          srcId: current,
          tgtId: next,
          shape,
          loopIdx: currentLoop,
          totalLoops: repeatCount,
        });
        current = next;
      }

      path.current = segments;

      // Start after layout settles
      setTimeout(() => {
        if (!playing.current) return;
        startPath();
        rafRef.current = requestAnimationFrame(render);
      }, 400);
    },
    [stop, render]
  );

  useImperativeHandle(ref, () => ({ play, stop }));

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        zIndex: 50,
      }}
    />
  );
});

function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r}, ${g}, ${b}`;
}
