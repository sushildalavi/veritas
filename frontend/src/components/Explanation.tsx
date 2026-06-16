export default function Explanation() {
  return (
    <div>
      <div className="page-header">
        <div className="hero-title">Explanation</div>
        <div className="page-desc">
          How Veritas generates citation-backed explanations using a fine-tuned Qwen2.5-1.5B via
          MLX LoRA on Apple Silicon.
        </div>
      </div>

      <div className="section">
        <div className="section-label">Explanation pipeline</div>
        <div className="pipeline-viz">
          {[
            { step: "01", title: "Claim + Evidence", desc: "Top-K BM25 passages are assembled with the original claim into a structured prompt." },
            { step: "02", title: "Citation injection", desc: "Each passage is labelled [1], [2], … [N] so the model can reference them inline." },
            { step: "03", title: "MLX LoRA adapter", desc: "Qwen2.5-1.5B-Instruct-4bit with a LoRA rank-8 adapter fine-tuned on FEVER-style reasoning." },
            { step: "04", title: "Citation validation", desc: "All [N] markers in the output are checked against the evidence set. citation_valid = true if all cited indices exist." },
            { step: "05", title: "Fallback", desc: "If the LoRA adapter is unavailable, Veritas falls back to a template-based explanation derived from NLI predictions." },
          ].map((s) => (
            <div key={s.step} className="pv-step">
              <div className="pv-num">{s.step}</div>
              <div className="pv-body">
                <div className="pv-title">{s.title}</div>
                <div className="pv-desc">{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-label">MLX LoRA training summary</div>
        <div className="metrics-table-wrap">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Base model",         "Qwen2.5-1.5B-Instruct-4bit"],
                ["Fine-tune method",   "LoRA (rank 8, alpha 16)"],
                ["Training hardware",  "Apple Silicon (MLX)"],
                ["Iterations",         "500"],
                ["Citation presence",  "0.72 @ 500 iters"],
                ["Training loss",      "1.8421 → 1.3102"],
                ["Format compliance",  "improved — structured reasoning format"],
                ["Adapter size",       "~15 MB (not committed to repo)"],
              ].map(([k, v]) => (
                <tr key={k as string}><td>{k}</td><td className="mono">{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <div className="section-label">Citation format example</div>
        <div className="answer-box">
          <div className="answer-text" style={{ lineHeight: "1.9" }}>
            The Apollo 11 mission did land humans on the Moon in 1969{" "}
            <sup className="cite-marker">1</sup>. Commander Neil Armstrong and lunar module pilot
            Buzz Aldrin became the first humans on the lunar surface on July 20, 1969{" "}
            <sup className="cite-marker">2</sup>. The lunar module Eagle landed in the Sea of
            Tranquility{" "}
            <sup className="cite-marker">3</sup>.
          </div>
          <div className="citation-chips">
            {[
              [1, "NASA Technical Reports"],
              [2, "Encyclopædia Britannica"],
              [3, "Wikipedia"],
            ].map(([id, src]) => (
              <div key={id as number} className="citation-chip">
                <span className="chip-num">[{id}]</span>
                <span>{src}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="uncertainty-panel">
        <div className="up-title">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M8 1.5L1.5 13.5h13L8 1.5z"/><path d="M8 6v4M8 11.5v.5"/>
          </svg>
          Limitations
        </div>
        <div className="up-item"><div className="up-dot"/><span>citation_presence of 0.72 means ~28% of outputs miss inline citations — not production-grade.</span></div>
        <div className="up-item"><div className="up-dot"/><span>500 iterations is a short run. A full fine-tune on a larger FEVER split would take far longer.</span></div>
        <div className="up-item"><div className="up-dot"/><span>The LoRA adapter is not committed to this repo (size). It runs locally only if <code className="code">adapters/</code> is present.</span></div>
      </div>
    </div>
  );
}
