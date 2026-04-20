import { useState, useEffect, useRef, useCallback } from "react";
import { WSClient } from "../lib/websocket-client";
import type { MetricSnapshot, FlowSequence, WSMessage } from "../types";

const WS_URL = `ws://${window.location.host}/ws/metrics`;

export function useMetricStream() {
  const [metrics, setMetrics] = useState<MetricSnapshot | null>(null);
  const [flow, setFlow] = useState<FlowSequence | null>(null);
  const [connected, setConnected] = useState(false);
  const clientRef = useRef<WSClient | null>(null);

  useEffect(() => {
    const client = new WSClient(WS_URL);
    clientRef.current = client;

    const unsub = client.subscribe((msg: WSMessage) => {
      if (msg.type === "metrics") {
        setMetrics(msg.data as MetricSnapshot);
      } else if (msg.type === "flow") {
        setFlow(msg.data as FlowSequence);
      }
    });

    client.connect();

    const checkInterval = setInterval(() => {
      setConnected(client.connected);
    }, 1000);

    return () => {
      unsub();
      clearInterval(checkInterval);
      client.disconnect();
    };
  }, []);

  const configure = useCallback(
    (config: Record<string, unknown>) => {
      clientRef.current?.send(config);
    },
    []
  );

  return { metrics, flow, connected, configure };
}
