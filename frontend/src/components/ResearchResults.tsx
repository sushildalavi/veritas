export default function ResearchResults() {
  return (
    <div>
      <div className="section">
        <h2>Research Results</h2>
        <p>
          All numbers measured on this hardware. Source files in <code className="code">reports/</code>.
          No results are fabricated.
        </p>
      </div>

      <div className="section">
        <h3>Verifier — Oracle vs Retrieved (650-example test set)</h3>
        <p style={{ marginBottom: "8px", fontSize: "12px" }}>
          Source: <code className="code">reports/oracle_vs_retrieved_v2_full.json</code> ·
          Checkpoint: <code className="code">checkpoints/transformer_verifier_clean</code>
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Mode</th><th>Evidence</th><th>Accuracy</th><th>Macro-F1</th><th>NEI FPR</th></tr>
            </thead>
            <tbody>
              <tr><td>bundle</td><td>oracle</td><td>0.7354</td><td>0.7246</td><td>0.0</td></tr>
              <tr className="bold-row"><td>per_passage_max</td><td>oracle</td><td>0.6985</td><td>0.6728</td><td>0.0</td></tr>
              <tr><td>bundle</td><td>retrieved</td><td>0.3600</td><td>0.3332</td><td>0.8839</td></tr>
              <tr className="bold-row"><td>per_passage_max</td><td>retrieved</td><td>0.4062</td><td>0.3887</td><td>0.7098</td></tr>
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: "12px", marginTop: "6px" }}>
          Oracle→retrieved gap (per_passage_max): accuracy −0.2923 · macro-F1 −0.2841
        </p>
      </div>

      <div className="section">
        <h3>Retrieval Profile Comparison (650 examples)</h3>
        <p style={{ marginBottom: "8px", fontSize: "12px" }}>
          Source: <code className="code">reports/retrieval_profile_comparison_650.json</code>
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Profile</th><th>recall@10</th><th>nDCG@10</th><th>Verifier macro-F1</th></tr>
            </thead>
            <tbody>
              <tr className="bold-row">
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
        <div className="note">
          Better retrieval recall does not monotonically improve verifier macro-F1.
          The sentence-transformer hybrid has higher recall@10 but lower end-to-end F1.
        </div>
      </div>

      <div className="section">
        <h3>Error Analysis (650 examples)</h3>
        <p style={{ marginBottom: "8px", fontSize: "12px" }}>
          Source: <code className="code">reports/error_analysis_650.json</code>
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Failure Bucket</th><th>Count</th><th>%</th></tr>
            </thead>
            <tbody>
              <tr className="bold-row"><td>oracle_correct_retrieved_wrong</td><td>222</td><td>34.15%</td></tr>
              <tr><td>oracle_wrong_retrieved_wrong</td><td>164</td><td>25.23%</td></tr>
              <tr><td>oracle_correct_retrieved_correct</td><td>214</td><td>32.92%</td></tr>
              <tr><td>oracle_wrong_retrieved_correct</td><td>50</td><td>7.69%</td></tr>
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: "12px", marginTop: "6px" }}>
          FEVER retrieved macro-F1: 0.4392 · SciFact retrieved macro-F1: 0.1817
        </p>
      </div>

      <div className="section">
        <h3>Inference Benchmark</h3>
        <p style={{ marginBottom: "8px", fontSize: "12px" }}>
          Source: <code className="code">reports/verifier_inference_benchmark.json</code>,
          <code className="code">reports/onnx_verifier_benchmark.json</code>
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Runtime</th><th>Batch</th><th>Device</th><th>Latency ms</th><th>Throughput ex/s</th></tr>
            </thead>
            <tbody>
              <tr><td>transformers (native)</td><td>1</td><td>CPU</td><td>16.0</td><td>62.4</td></tr>
              <tr><td>transformers (native)</td><td>8</td><td>CPU</td><td>58.9</td><td>135.8</td></tr>
              <tr className="bold-row"><td>transformers (native)</td><td>1</td><td>MPS</td><td>6.5</td><td>153.8</td></tr>
              <tr className="bold-row"><td>transformers (native)</td><td>8</td><td>MPS</td><td>23.5</td><td>340.9</td></tr>
              <tr><td>ONNX (CPU only)</td><td>1</td><td>CPU</td><td>18.2</td><td>55.1</td></tr>
              <tr><td>ONNX (CPU only)</td><td>8</td><td>CPU</td><td>133.9</td><td>59.7</td></tr>
            </tbody>
          </table>
        </div>
        <div className="note">
          ONNX is slower than native transformers on this Mac. MPS via transformers
          (153 ex/s) is 3× faster than ONNX CPU. ONNX export is valid; speedup applies on CUDA.
        </div>
      </div>

      <div className="section">
        <h3>Resume-Safe Summary</h3>
        <div className="card">
          <p style={{ fontSize: "13px", lineHeight: "1.75", color: "var(--text)" }}>
            Built Veritas, a failure-aware evidence-grounded fact-verification system with BM25
            retrieval, DistilRoBERTa NLI verification, oracle-vs-retrieved ablations,
            ONNX/MLX inference benchmarking, SFT/DPO training-data generation, FastAPI REST APIs,
            and a React TypeScript research dashboard; measured a{" "}
            <strong>0.6728 oracle macro-F1 vs. 0.3887 retrieved macro-F1 gap</strong> and documented
            retrieval/noisy-evidence bottlenecks through full-set error analysis; improved
            MLX LoRA citation compliance from 0.10 to 0.72.
          </p>
        </div>
        <div className="note" style={{ marginTop: "12px" }}>
          Do not claim: ONNX faster than native on Mac · robust verifier or gate improved macro-F1
          · Phi-3 QLoRA/DPO adapter trained · MLX explanation adapter is production-grade.
        </div>
      </div>
    </div>
  );
}
