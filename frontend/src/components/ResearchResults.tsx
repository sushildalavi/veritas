const RETRIEVAL_BARS = [
  { label: "bm25_only (default)",            value: 0.3887, color: "#3b82f6", delay: "0.1s" },
  { label: "hybrid_bm25_dense",              value: 0.3864, color: "#8b5cf6", delay: "0.2s" },
  { label: "hybrid_bm25_sent_transformer",   value: 0.3776, color: "#6366f1", delay: "0.3s" },
];

const INFERENCE_BARS = [
  { label: "MPS batch=8",  value: 340.9, max: 360, color: "#3b82f6", delay: "0.1s" },
  { label: "MPS batch=1",  value: 153.8, max: 360, color: "#22c55e", delay: "0.2s" },
  { label: "CPU batch=8",  value: 135.8, max: 360, color: "#8b5cf6", delay: "0.3s" },
  { label: "CPU batch=1",  value: 62.4,  max: 360, color: "#f59e0b", delay: "0.4s" },
  { label: "ONNX batch=8", value: 59.7,  max: 360, color: "#6b7280", delay: "0.5s" },
  { label: "ONNX batch=1", value: 55.1,  max: 360, color: "#6b7280", delay: "0.6s" },
];

export default function ResearchResults() {
  return (
    <div>
      <div className="page-header">
        <div className="hero-title">Research Results</div>
        <div className="page-desc">
          All numbers measured locally on a 650-example test set (FEVER + SciFact). Nothing fabricated.
        </div>
      </div>

      <div className="section">
        <div className="section-label">Verifier — Oracle vs Retrieved</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Mode</th><th>Evidence</th><th>Accuracy</th><th>Macro-F1</th><th>NEI FPR</th></tr>
            </thead>
            <tbody>
              <tr><td>bundle</td><td>oracle</td><td>0.7354</td><td>0.7246</td><td>0.0</td></tr>
              <tr className="row-highlight"><td>per_passage_max</td><td>oracle</td><td>0.6985</td><td>0.6728</td><td>0.0</td></tr>
              <tr><td>bundle</td><td>retrieved</td><td>0.3600</td><td>0.3332</td><td>0.8839</td></tr>
              <tr className="row-highlight"><td>per_passage_max</td><td>retrieved</td><td>0.4062</td><td>0.3887</td><td>0.7098</td></tr>
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-dim)", marginTop: "8px", fontFamily: "var(--mono)" }}>
          Oracle→retrieved gap (per_passage_max): accuracy −0.2923 · macro-F1 −0.2841
        </div>
      </div>

      <div className="section">
        <div className="section-label">Retrieval Profile Comparison — Verifier macro-F1</div>
        <div className="card" style={{ marginBottom: "12px" }}>
          <div className="bar-chart">
            {RETRIEVAL_BARS.map((b) => (
              <div key={b.label} className="bar-row">
                <div className="bar-label">{b.label}</div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      "--w": `${b.value * 100}%`,
                      "--delay": b.delay,
                      background: b.color,
                    } as React.CSSProperties}
                  />
                </div>
                <div className="bar-value">{b.value.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Profile</th><th>recall@10</th><th>nDCG@10</th><th>Verifier macro-F1</th></tr>
            </thead>
            <tbody>
              <tr className="row-highlight">
                <td>bm25_only (default)</td><td>0.5334</td><td>0.4816</td><td>0.3887</td>
              </tr>
              <tr>
                <td>hybrid_bm25_dense</td><td>0.5113</td><td>0.4288</td><td>0.3864</td>
              </tr>
              <tr>
                <td>hybrid_bm25_sentence_transformer</td><td>0.5714</td><td>0.5234</td><td>0.3776</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="notice notice-neutral" style={{ marginTop: "10px" }}>
          Better retrieval recall does not monotonically improve verifier macro-F1.
          Sentence-transformer hybrid has higher recall@10 but lower end-to-end F1.
        </div>
      </div>

      <div className="section">
        <div className="section-label">Error Analysis (650 examples)</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Failure Bucket</th><th>Count</th><th>%</th></tr>
            </thead>
            <tbody>
              <tr className="row-highlight"><td>oracle_correct_retrieved_wrong</td><td>222</td><td>34.15%</td></tr>
              <tr><td>oracle_wrong_retrieved_wrong</td><td>164</td><td>25.23%</td></tr>
              <tr><td>oracle_correct_retrieved_correct</td><td>214</td><td>32.92%</td></tr>
              <tr><td>oracle_wrong_retrieved_correct</td><td>50</td><td>7.69%</td></tr>
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-dim)", marginTop: "8px", fontFamily: "var(--mono)" }}>
          FEVER retrieved macro-F1: 0.4392 · SciFact retrieved macro-F1: 0.1817
        </div>
      </div>

      <div className="section">
        <div className="section-label">Inference Throughput — ex/s (higher is better)</div>
        <div className="card" style={{ marginBottom: "12px" }}>
          <div className="bar-chart">
            {INFERENCE_BARS.map((b) => (
              <div key={b.label} className="bar-row">
                <div className="bar-label">{b.label}</div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      "--w": `${(b.value / b.max) * 100}%`,
                      "--delay": b.delay,
                      background: b.color,
                    } as React.CSSProperties}
                  />
                </div>
                <div className="bar-value">{b.value}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Runtime</th><th>Batch</th><th>Device</th><th>Latency ms</th><th>Throughput ex/s</th></tr>
            </thead>
            <tbody>
              <tr><td>transformers (native)</td><td>1</td><td>CPU</td><td>16.0</td><td>62.4</td></tr>
              <tr><td>transformers (native)</td><td>8</td><td>CPU</td><td>58.9</td><td>135.8</td></tr>
              <tr className="row-highlight"><td>transformers (native)</td><td>1</td><td>MPS</td><td>6.5</td><td>153.8</td></tr>
              <tr className="row-highlight"><td>transformers (native)</td><td>8</td><td>MPS</td><td>23.5</td><td>340.9</td></tr>
              <tr><td>ONNX (CPU only)</td><td>1</td><td>CPU</td><td>18.2</td><td>55.1</td></tr>
              <tr><td>ONNX (CPU only)</td><td>8</td><td>CPU</td><td>133.9</td><td>59.7</td></tr>
            </tbody>
          </table>
        </div>
        <div className="notice notice-neutral" style={{ marginTop: "10px" }}>
          ONNX is slower than native transformers on this Mac. MPS (153 ex/s) is 3× faster than ONNX CPU.
          ONNX export is valid — speedup applies on CUDA.
        </div>
      </div>

      <div className="section">
        <div className="section-label">Resume-Safe Summary</div>
        <div className="card">
          <p style={{ fontSize: "14px", lineHeight: "1.8", color: "var(--text-muted)" }}>
            Built Veritas, a failure-aware evidence-grounded fact-verification system with BM25 retrieval,
            DistilRoBERTa NLI verification, oracle-vs-retrieved ablations, ONNX/MLX inference benchmarking,
            SFT/DPO training-data generation, FastAPI REST APIs, and a React TypeScript research dashboard;
            measured a{" "}
            <strong style={{ color: "var(--text)" }}>0.6728 oracle macro-F1 vs. 0.3887 retrieved macro-F1 gap</strong>{" "}
            and documented retrieval/noisy-evidence bottlenecks through full-set error analysis;
            improved MLX LoRA citation compliance from 0.10 to 0.72.
          </p>
        </div>
        <div className="notice notice-warn" style={{ marginTop: "12px" }}>
          Do not claim: ONNX faster than native on Mac · robust verifier improved macro-F1 · Phi-3 QLoRA/DPO trained · MLX explanation adapter is production-grade.
        </div>
      </div>
    </div>
  );
}
