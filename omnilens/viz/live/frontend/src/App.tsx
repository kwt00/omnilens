import { useState, useEffect, useCallback } from "react";
import "./styles/global.css";

const API = "";

// --- Types ---

interface HeatmapCell {
  layer: number;
  position: number;
  token: string;
  norm: number;
}

interface HeatmapData {
  tokens: string[];
  n_layers: number;
  cells: HeatmapCell[];
  top_predictions: string[];
}

interface AttentionHead {
  head: number;
  weights: number[][];
}

interface AttentionData {
  layer: number;
  tokens: string[];
  heads: AttentionHead[];
}

interface FeatureActivation {
  feature_idx: number;
  activation: number;
}

interface TokenFeatures {
  position: number;
  token: string;
  features: FeatureActivation[];
}

interface FeaturesData {
  hook_point: string;
  tokens: string[];
  token_features: TokenFeatures[];
}

interface ModelInfo {
  model_name: string;
  n_layers: number;
  sae_attached: boolean;
  sae_hook_point?: string;
  sae_n_features?: number;
}

// --- Main App ---

export default function App() {
  const [text, setText] = useState("The capital of France is");
  const [heatmap, setHeatmap] = useState<HeatmapData | null>(null);
  const [attention, setAttention] = useState<AttentionData | null>(null);
  const [features, setFeatures] = useState<FeaturesData | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  // Load model info on mount
  useEffect(() => {
    fetch(`${API}/api/info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => {});
  }, []);

  const runModel = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setHeatmap(data);
      setSelectedLayer(Math.floor(data.n_layers / 2));

      // Load attention for the selected layer
      const attnRes = await fetch(`${API}/api/attention/${Math.floor(data.n_layers / 2)}`);
      setAttention(await attnRes.json());

      // Load features if SAE attached
      if (info?.sae_attached) {
        const featRes = await fetch(`${API}/api/features`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        setFeatures(await featRes.json());
      }
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }, [text, info]);

  const loadAttention = useCallback(async (layer: number) => {
    setSelectedLayer(layer);
    try {
      const res = await fetch(`${API}/api/attention/${layer}`);
      setAttention(await res.json());
    } catch (e) {
      console.error(e);
    }
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>OmniLens</h1>
        {info && (
          <span className="header-info">
            {info.model_name} | {info.n_layers} layers
            {info.sae_attached && ` | SAE: ${info.sae_n_features} features`}
          </span>
        )}
      </header>

      <div className="input-bar">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runModel()}
          placeholder="Enter text and press Enter..."
        />
        <button onClick={runModel} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {heatmap && (
        <>
          {/* Top predictions */}
          <div className="predictions">
            Top predictions: {heatmap.top_predictions.map((p, i) => (
              <span key={i} className="pred-token">{p}</span>
            ))}
          </div>

          <div className="panels">
            {/* Panel 1: Activation Heatmap */}
            <div className="panel heatmap-panel">
              <h2>Activation Heatmap</h2>
              <ActivationHeatmap
                data={heatmap}
                selectedLayer={selectedLayer}
                selectedToken={selectedToken}
                onSelectLayer={loadAttention}
                onSelectToken={setSelectedToken}
              />
            </div>

            {/* Panel 2: Attention Explorer */}
            <div className="panel attention-panel">
              <h2>
                Attention — Layer {selectedLayer}
                <select
                  value={selectedLayer}
                  onChange={(e) => loadAttention(Number(e.target.value))}
                  className="layer-select"
                >
                  {Array.from({ length: heatmap.n_layers }, (_, i) => (
                    <option key={i} value={i}>Layer {i}</option>
                  ))}
                </select>
              </h2>
              {attention && (
                <AttentionExplorer
                  data={attention}
                  selectedToken={selectedToken}
                  onSelectToken={setSelectedToken}
                />
              )}
            </div>
          </div>

          {/* Panel 3: Feature Dashboard */}
          {features && (
            <div className="panel feature-panel">
              <h2>SAE Features — {features.hook_point}</h2>
              <FeatureDashboard
                data={features}
                selectedToken={selectedToken}
                onSelectToken={setSelectedToken}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

// --- Panel 1: Activation Heatmap ---

function ActivationHeatmap({
  data,
  selectedLayer,
  selectedToken,
  onSelectLayer,
  onSelectToken,
}: {
  data: HeatmapData;
  selectedLayer: number;
  selectedToken: number | null;
  onSelectLayer: (l: number) => void;
  onSelectToken: (t: number | null) => void;
}) {
  const { tokens, n_layers, cells } = data;

  // Find max norm for color scaling
  const maxNorm = Math.max(...cells.map((c) => c.norm), 1e-8);

  // Build grid: cells[layer * n_tokens + pos]
  const grid: number[][] = [];
  for (let l = 0; l < n_layers; l++) {
    const row: number[] = [];
    for (let p = 0; p < tokens.length; p++) {
      const cell = cells.find((c) => c.layer === l && c.position === p);
      row.push(cell ? cell.norm / maxNorm : 0);
    }
    grid.push(row);
  }

  return (
    <div className="heatmap-container">
      {/* Token labels (top) */}
      <div className="heatmap-header">
        <div className="heatmap-corner" />
        {tokens.map((t, i) => (
          <div
            key={i}
            className={`heatmap-token ${selectedToken === i ? "selected" : ""}`}
            onClick={() => onSelectToken(selectedToken === i ? null : i)}
          >
            {t}
          </div>
        ))}
      </div>

      {/* Heatmap rows */}
      <div className="heatmap-body">
        {grid.map((row, layer) => (
          <div
            key={layer}
            className={`heatmap-row ${selectedLayer === layer ? "selected" : ""}`}
            onClick={() => onSelectLayer(layer)}
          >
            <div className="heatmap-label">L{layer}</div>
            {row.map((val, pos) => (
              <div
                key={pos}
                className={`heatmap-cell ${selectedToken === pos ? "highlight" : ""}`}
                style={{
                  backgroundColor: `rgba(59, 130, 246, ${val})`,
                }}
                title={`Layer ${layer}, "${tokens[pos]}": ${(val * maxNorm).toFixed(2)}`}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Panel 2: Attention Explorer ---

function AttentionExplorer({
  data,
  selectedToken,
  onSelectToken,
}: {
  data: AttentionData;
  selectedToken: number | null;
  onSelectToken: (t: number | null) => void;
}) {
  const [selectedHead, setSelectedHead] = useState(0);
  const { tokens, heads } = data;

  if (heads.length === 0) return <p>No attention data</p>;

  const head = heads[Math.min(selectedHead, heads.length - 1)];
  const weights = head.weights;
  const maxWeight = Math.max(...weights.flat(), 1e-8);

  return (
    <div className="attention-container">
      {/* Head selector */}
      <div className="head-selector">
        {heads.map((h) => (
          <button
            key={h.head}
            className={`head-btn ${selectedHead === h.head ? "active" : ""}`}
            onClick={() => setSelectedHead(h.head)}
          >
            H{h.head}
          </button>
        ))}
      </div>

      {/* Attention matrix */}
      <div className="attention-matrix">
        {/* Column headers (key tokens) */}
        <div className="attn-header">
          <div className="attn-corner" />
          {tokens.map((t, i) => (
            <div key={i} className="attn-token">{t}</div>
          ))}
        </div>

        {/* Rows (query tokens) */}
        {weights.map((row, qi) => (
          <div
            key={qi}
            className={`attn-row ${selectedToken === qi ? "selected" : ""}`}
            onClick={() => onSelectToken(selectedToken === qi ? null : qi)}
          >
            <div className="attn-label">{tokens[qi]}</div>
            {row.map((w, ki) => (
              <div
                key={ki}
                className="attn-cell"
                style={{
                  backgroundColor: `rgba(234, 88, 12, ${w / maxWeight})`,
                }}
                title={`"${tokens[qi]}" → "${tokens[ki]}": ${w.toFixed(4)}`}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Panel 3: Feature Dashboard ---

function FeatureDashboard({
  data,
  selectedToken,
  onSelectToken,
}: {
  data: FeaturesData;
  selectedToken: number | null;
  onSelectToken: (t: number | null) => void;
}) {
  const { token_features } = data;
  const maxAct = Math.max(
    ...token_features.flatMap((tf) => tf.features.map((f) => f.activation)),
    1e-8
  );

  return (
    <div className="features-container">
      <div className="features-tokens">
        {token_features.map((tf) => (
          <div
            key={tf.position}
            className={`feature-token-col ${selectedToken === tf.position ? "selected" : ""}`}
            onClick={() => onSelectToken(selectedToken === tf.position ? null : tf.position)}
          >
            <div className="feature-token-label">{tf.token}</div>
            <div className="feature-bars">
              {tf.features.slice(0, 10).map((f) => (
                <div key={f.feature_idx} className="feature-bar-row">
                  <span className="feature-idx">#{f.feature_idx}</span>
                  <div className="feature-bar-bg">
                    <div
                      className="feature-bar-fill"
                      style={{ width: `${(f.activation / maxAct) * 100}%` }}
                    />
                  </div>
                  <span className="feature-val">{f.activation.toFixed(2)}</span>
                </div>
              ))}
              {tf.features.length === 0 && (
                <span className="feature-none">none</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
