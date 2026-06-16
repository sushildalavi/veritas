import { useState } from "react";
import { runPipeline } from "../api";
import type { PipelineResponse } from "../types";

function verdictClass(v: string) {
  if (v === "SUPPORTED") return "supported";
  if (v === "REFUTED") return "refuted";
  return "nei";
}

export default function VerifyClaim() {
  const [claim, setClaim] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    if (!claim.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await runPipeline(claim.trim(), topK);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="section">
        <h2>Verify a Claim</h2>
        <p>Runs the full pipeline: BM25 retrieval → NLI verification → explanation generation.</p>
      </div>

      <div className="input-group vertical">
        <textarea
          placeholder="Enter a claim to verify, e.g. 'Marie Curie won the Nobel Prize in Physics.'"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleRun();
          }}
        />
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <div className="topk-row">
            <label>Top-K evidence:</label>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <button className="btn" onClick={handleRun} disabled={loading || !claim.trim()}>
            {loading ? <><span className="spinner" />Running…</> : "Run full pipeline"}
          </button>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="row">
            <span className="key">Verdict</span>
            <span className={`verdict-badge ${verdictClass(result.verdict)}`}>{result.verdict}</span>
          </div>
          <div className="row">
            <span className="key">Confidence</span>
            <span className="val">{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="row">
            <span className="key">Backend</span>
            <span className="val">{result.backend_used} / {result.retrieval_backend}</span>
          </div>
          <div className="row">
            <span className="key">Citations valid</span>
            <span className="val">{result.citation_valid ? "✓" : "✗"}</span>
          </div>

          <hr className="divider" />

          <h3>Explanation</h3>
          <div className="explanation-text">{result.explanation || <em style={{ color: "var(--text-dim)" }}>No explanation generated.</em>}</div>

          {result.citations.length > 0 && (
            <div style={{ marginTop: "10px", fontSize: "12px", color: "var(--text-muted)" }}>
              Citations: {result.citations.join(", ")}
            </div>
          )}

          <hr className="divider" />

          <h3>Retrieved Evidence ({result.evidence.length})</h3>
          <div className="evidence-list">
            {result.evidence.map((ev, i) => (
              <div key={ev.doc_id + i} className="evidence-item">
                <div className="ev-header">
                  <span className="ev-id">E{ev.citation_id ?? i + 1}</span>
                  {ev.title && <span className="ev-title">{ev.title}</span>}
                  {ev.score != null && <span className="ev-score">{ev.score.toFixed(3)}</span>}
                </div>
                <div className="ev-text">{ev.text}</div>
              </div>
            ))}
          </div>

          <hr className="divider" />

          <h3>Latency Breakdown</h3>
          <div className="latency-row">
            <div className="latency-chip">retrieval<span>{result.latency.retrieval_ms.toFixed(1)} ms</span></div>
            <div className="latency-chip">verification<span>{result.latency.verification_ms.toFixed(1)} ms</span></div>
            <div className="latency-chip">explanation<span>{result.latency.explanation_ms.toFixed(1)} ms</span></div>
            <div className="latency-chip">total<span>{result.latency.total_ms.toFixed(1)} ms</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
