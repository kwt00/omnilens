import { useState, useEffect } from "react";
import type { GraphResponse } from "../types";

const API_BASE = window.location.origin;

export function useGraphData() {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchGraph() {
      try {
        const res = await fetch(`${API_BASE}/api/graph`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: GraphResponse = await res.json();
        if (!cancelled) {
          setGraph(data);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch graph");
          setLoading(false);
        }
      }
    }

    fetchGraph();
    return () => { cancelled = true; };
  }, []);

  return { graph, loading, error };
}
