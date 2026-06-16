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
      <div className="page-header">
        <div className="page-title">Verify Claim</div>
        <div className="page-desc">
          Runs the full pipeline: BM25 retrieval → NLI verification → explanation generation.
          Press <code className="code">⌘ Enter</code> to submit.
        </div>
      </div>

      <div className="form-group">
        <textarea
          placeholder="Enter a claim to verify, e.g. 'Marie Curie won the Nobel Prize in Physics.'"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleRun(); }}
        />
        <div className="input-row">
          <div className="topk-row">
            <label>Top-K evidence:</label>
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[1, 2, 3, 5, 10, 20].map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>
          <button className="btn" onClick={handleRun} disabled={loading || !claim.trim()}>
            {loading ? <><span className="spinner" />Running…</> : "Run pipeline"}
          </button>
        </div>
      </div>

      {error && <div className="notice notice-error">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="result-top">
            <div className="result-verdict-block">
              <div className="result-verdict-label">Verdict</div>
              <span className={`verdict ${verdictClass(result.verdict)}`}>{result.verdict}</span>
            </div>
            <div className="result-meta">
              <div className="result-meta-item">
                <div className="result-meta-label">Confidence</div>
                <div className="result-meta-value">{(result.confidence * 100).toFixed(1)}%</div>
              </div>
              <div className="result-meta-item">
                <div className="result-meta-label">Verifier</div>
                <div className="result-meta-value">{result.backend_used}</div>
              </div>
              <div className="result-meta-item">
                <div className="result-meta-label">Retrieval</div>
                <div className="result-meta-value">{result.retrieval_backend}</div>
              </div>
              <div className="result-meta-item">
                <div className="result-meta-label">Citations</div>
                <div className="result-meta-value" style={{ color: result.citation_valid ? "var(--green)" : "var(--text-dim)" }}>
                  {result.citation_valid ? "valid" : "none"}
                </div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <div className="result-section-title">Explanation</div>
            <div className="explanation-box">
              {result.explanation || <em style={{ color: "var(--text-dim)" }}>No explanation generated.</em>}
            </div>
            {result.citations.length > 0 && (
              <div style={{ marginTop: "8px", fontSize: "12px", color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
                Citations: {result.citations.join(", ")}
              </div>
            )}
          </div>

          <div style={{ marginBottom: "20px" }}>
            <div className="result-section-title">Retrieved Evidence ({result.evidence.length})</div>
            <div className="evidence-list">
              {result.evidence.map((ev, i) => (
                <div key={ev.doc_id + i} className="evidence-item">
                  <div className="evidence-header">
                    <span className="ev-num">E{ev.citation_id ?? i + 1}</span>
                    {ev.title && <span className="ev-title">{ev.title}</span>}
                    {ev.score != null && <span className="ev-score">{ev.score.toFixed(3)}</span>}
                  </div>
                  <div className="ev-text">{ev.text}</div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="result-section-title">Latency</div>
            <div className="latency-row">
              <div className="latency-chip">retrieval<span>{result.latency.retrieval_ms.toFixed(1)} ms</span></div>
              <div className="latency-chip">verification<span>{result.latency.verification_ms.toFixed(1)} ms</span></div>
              <div className="latency-chip">explanation<span>{result.latency.explanation_ms.toFixed(1)} ms</span></div>
              <div className="latency-chip">total<span>{result.latency.total_ms.toFixed(1)} ms</span></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
