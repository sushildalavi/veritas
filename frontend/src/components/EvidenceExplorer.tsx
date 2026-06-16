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
      <div className="page-header">
        <div className="hero-title">Evidence Explorer</div>
        <div className="page-desc">
          Retrieve evidence passages for a claim without running the verifier.
        </div>
      </div>

      <div className="form-group">
        <input
          type="text"
          placeholder="Enter a claim, e.g. 'The Eiffel Tower is in Paris.'"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleRetrieve()}
        />
        <div className="input-row">
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

      {error && <div className="notice notice-error">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="result-top" style={{ marginBottom: "16px", paddingBottom: "14px" }}>
            <div className="result-meta">
              <div className="result-meta-item">
                <div className="result-meta-label">Backend</div>
                <div className="result-meta-value">{result.retrieval_backend}</div>
              </div>
              <div className="result-meta-item">
                <div className="result-meta-label">Latency</div>
                <div className="result-meta-value">{result.latency_ms.toFixed(1)} ms</div>
              </div>
              <div className="result-meta-item">
                <div className="result-meta-label">Passages</div>
                <div className="result-meta-value">{result.evidence.length}</div>
              </div>
            </div>
          </div>

          <div className="evidence-list">
            {result.evidence.length === 0 && (
              <div style={{ color: "var(--text-dim)", fontSize: "13px", padding: "8px 0" }}>
                No evidence retrieved.
              </div>
            )}
            {result.evidence.map((ev, i) => (
              <div key={ev.doc_id + i} className="evidence-item">
                <div className="evidence-header">
                  <span className="ev-num">#{ev.citation_id ?? i + 1}</span>
                  {ev.title && <span className="ev-title">{ev.title}</span>}
                  {ev.score != null && <span className="ev-score">{ev.score.toFixed(4)}</span>}
                </div>
                <div className="ev-text">{ev.text}</div>
                <div className="ev-docid">{ev.doc_id}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
