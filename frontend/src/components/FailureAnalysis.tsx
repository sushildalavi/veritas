import { CountUp } from "./CountUp";

export default function FailureAnalysis() {
  const barData = [
    { label: "Oracle macro-F1",     value: 0.6728, cls: "bar-blue" },
    { label: "Retrieved macro-F1",  value: 0.3887, cls: "bar-amber" },
    { label: "Robust retrain F1",   value: 0.2829, cls: "bar-red"   },
    { label: "Recall@10",           value: 0.5334, cls: "bar-blue"  },
  ];

  return (
    <div>
      <div className="page-header">
        <div className="hero-title">Failure Analysis</div>
        <div className="page-desc">
          Honest negative results, retrieval bottlenecks, and what Veritas cannot do.
        </div>
      </div>

      {/* Oracle vs Retrieved gap */}
      <div className="section">
        <div className="section-label">Oracle vs retrieved — the gap</div>
        <div className="bar-chart">
          {barData.map((d) => (
            <div key={d.label} className="bar-row">
              <div className="bar-label">{d.label}</div>
              <div className="bar-track">
                <div className={`bar-fill ${d.cls}`} style={{ width: `${d.value * 100}%` }} />
              </div>
              <div className="bar-val mono">
                <CountUp to={d.value} decimals={4} duration={900} />
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: "12px", fontSize: "12px", color: "var(--text-dim)" }}>
          Oracle macro-F1 measures performance when the verifier receives ground-truth evidence.
          The 0.2841 gap is caused entirely by BM25 retrieval failure — the verifier itself is
          strong when it has the right evidence.
        </div>
      </div>

      {/* Negative results table */}
      <div className="section">
        <div className="section-label">Negative results</div>
        <div className="metrics-table-wrap">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Result</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Robust verifier retrain",    "macro-F1 0.2829",  "Regressed vs. clean verifier. Likely label noise or insufficient data."],
                ["Relevance gate filter",      "No improvement",   "Filtering by retrieval score threshold did not lift retrieved F1."],
                ["Dense retrieval (not tried)","N/A",              "BM25-only system. DPR/ColBERT not yet integrated."],
                ["LoRA at 500 iters",          "citation 0.72",    "Short run — not production-ready. Would need 2000+ iters minimum."],
                ["DPO alignment",              "Not trained",      "DPO was proposed but not executed in this study."],
              ].map(([exp, res, note]) => (
                <tr key={exp as string}>
                  <td>{exp}</td>
                  <td className="mono" style={{ color: "var(--amber)" }}>{res}</td>
                  <td style={{ color: "var(--text-dim)", fontSize: "12px" }}>{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* System limitations */}
      <div className="section">
        <div className="section-label">System limitations</div>
        <div className="grid-2">
          {[
            { title: "Retrieval is the bottleneck", body: "BM25 Recall@10 of 0.5334 means the verifier never sees relevant evidence for ~47% of claims." },
            { title: "Single-hop reasoning only",   body: "Multi-hop claims requiring inference across multiple documents are not supported." },
            { title: "FEVER distribution",          body: "Trained and evaluated on FEVER. Performance on claims outside the Wikipedia knowledge base is unknown." },
            { title: "Static knowledge base",       body: "Evidence corpus is a fixed Wikipedia snapshot. No live retrieval from the web." },
            { title: "NLI ≠ fact-checking",        body: "DistilRoBERTa NLI predicts entailment, not factual accuracy. These are not the same." },
            { title: "No calibrated confidence",    body: "Softmax probabilities are not calibrated. Stated confidence should not be taken as a probability." },
          ].map((c) => (
            <div key={c.title} className="info-card">
              <div className="ic-title">{c.title}</div>
              <div className="ic-body">{c.body}</div>
            </div>
          ))}
        </div>
      </div>

      {/* What Veritas Is Not */}
      <div className="section">
        <div className="section-label">What Veritas is not</div>
        <div className="uncertainty-panel" style={{ marginTop: 0 }}>
          <div className="up-title">Hard constraints — do not over-claim</div>
          {[
            "Not a production fact-checker. It is a research prototype evaluated on FEVER.",
            "Not deployed to any cloud service. It runs locally on port 8001.",
            "Not SOTA. Oracle macro-F1 0.6728 is reasonable for a clean NLI verifier but is not state-of-the-art.",
            "Not using GPT, Claude, or any large API-based LLM for verification. The verifier is DistilRoBERTa.",
            "The robust verifier and relevance gate did not improve metrics — those are negative results, not successes.",
            "MLX LoRA training reached citation_presence 0.72 after 500 iters on Apple Silicon. This is early-stage.",
          ].map((item) => (
            <div key={item} className="up-item"><div className="up-dot"/><span>{item}</span></div>
          ))}
        </div>
      </div>
    </div>
  );
}
