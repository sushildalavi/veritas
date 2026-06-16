import { useState } from "react";
import { runRetrieve } from "../api";
import type { RetrieveResponse } from "../types";

export default function EvidenceExplorer() {
  const [claim, setClaim] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRetrieve() {
    if (!claim.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await runRetrieve(claim.trim(), topK);
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
        <h2>Evidence Explorer</h2>
        <p>Retrieve evidence passages for a claim without running the full verifier.</p>
      </div>

      <div className="input-group vertical">
        <input
          type="text"
          placeholder="Enter a claim, e.g. 'The Eiffel Tower is in Paris.'"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleRetrieve()}
        />
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <div className="topk-row">
            <label>Top-K:</label>
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[1, 2, 3, 5, 10, 20].map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>
          <button className="btn" onClick={handleRetrieve} disabled={loading || !claim.trim()}>
            {loading ? <><span className="spinner" />Retrieving…</> : "Retrieve evidence"}
          </button>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="row">
            <span className="key">Backend</span>
            <span className="val">{result.retrieval_backend}</span>
          </div>
          <div className="row">
            <span className="key">Latency</span>
            <span className="val">{result.latency_ms.toFixed(1)} ms</span>
          </div>
          <div className="row">
            <span className="key">Results</span>
            <span className="val">{result.evidence.length} passages</span>
          </div>

          <hr className="divider" />

          <div className="evidence-list">
            {result.evidence.length === 0 && (
              <p style={{ color: "var(--text-muted)", fontSize: "13px" }}>No evidence retrieved.</p>
            )}
            {result.evidence.map((ev, i) => (
              <div key={ev.doc_id + i} className="evidence-item">
                <div className="ev-header">
                  <span className="ev-id">#{ev.citation_id ?? i + 1}</span>
                  {ev.title && <span className="ev-title">{ev.title}</span>}
                  {ev.score != null && <span className="ev-score">score: {ev.score.toFixed(4)}</span>}
                </div>
                <div className="ev-text">{ev.text}</div>
                <div style={{ marginTop: "6px", fontSize: "11px", color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
                  {ev.doc_id}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
