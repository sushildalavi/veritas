export default function TrainingArtifacts() {
  return (
    <div>
      <div className="section">
        <h2>Training Artifacts</h2>
        <p>
          Veritas separates verdict classification from explanation generation.
          The production verifier is unchanged. All results below are measured.
        </p>
      </div>

      <div className="section">
        <h3>Artifact Status</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Artifact</th>
                <th>Status</th>
                <th>Path</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Production verifier</td>
                <td><span className="pos-badge">Production</span></td>
                <td><code className="code">checkpoints/transformer_verifier_clean</code></td>
                <td>DistilRoBERTa NLI; oracle macro-F1 0.6728</td>
              </tr>
              <tr>
                <td>SFT explanation dataset</td>
                <td><span className="pos-badge">Built</span></td>
                <td><code className="code">data/explanations/sft_*.jsonl</code></td>
                <td>256 training / val / test examples</td>
              </tr>
              <tr>
                <td>DPO preference dataset</td>
                <td><span className="pos-badge">Built</span></td>
                <td><code className="code">data/explanations/dpo_*.jsonl</code></td>
                <td>Synthetic rejection pairs</td>
              </tr>
              <tr>
                <td>MLX LoRA — verdict prediction</td>
                <td><span className="pos-badge">Trained</span></td>
                <td><code className="code">checkpoints/mlx_lora_verifier</code></td>
                <td>100 iters; acc 0.695; macro-F1 0.4632; 53.7 tok/s</td>
              </tr>
              <tr>
                <td>MLX LoRA — explanation SFT</td>
                <td><span className="pos-badge">Bug fixed + retrained</span></td>
                <td><code className="code">adapters/mlx_qwen_veritas_lora</code></td>
                <td>300 iters; format correctness 0.2 (partial)</td>
              </tr>
              <tr>
                <td>Phi-3 QLoRA adapter</td>
                <td><span className="blocked-badge">CUDA blocked</span></td>
                <td>—</td>
                <td>Script + config + Colab notebook ready; no CUDA on Mac</td>
              </tr>
              <tr>
                <td>Phi-3 DPO adapter</td>
                <td><span className="blocked-badge">CUDA blocked</span></td>
                <td>—</td>
                <td>Requires QLoRA adapter first; CUDA unavailable</td>
              </tr>
              <tr>
                <td>Robust verifier retrain</td>
                <td><span className="neg-badge">Negative result</span></td>
                <td><code className="code">checkpoints/transformer_verifier_robust</code></td>
                <td>Regressed: retrieved F1 0.3887→0.2829; not in production</td>
              </tr>
              <tr>
                <td>Relevance gate</td>
                <td><span className="neg-badge">Negative result</span></td>
                <td>In-memory (disabled)</td>
                <td>Improved NEI FPR (0.71→0.29) but regressed macro-F1 (0.3887→0.3557)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <h3>MLX LoRA — Verdict Prediction (measured)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
              <tr><td>verdict_accuracy</td><td>0.695</td></tr>
              <tr><td>macro_F1</td><td>0.4632</td></tr>
              <tr><td>SUPPORTED F1</td><td>0.7024</td></tr>
              <tr><td>REFUTED F1</td><td>0.6872</td></tr>
              <tr><td>NOT ENOUGH INFO F1</td><td>0.0 (rarely predicted)</td></tr>
              <tr><td>citation_valid_rate</td><td>0.60</td></tr>
              <tr><td>throughput</td><td>53.7 tok/s (Apple Silicon)</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <h3>MLX LoRA — Explanation SFT (measured at 300 iters)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Metric</th><th>Base model</th><th>Adapter (300 iters)</th></tr>
            </thead>
            <tbody>
              <tr><td>format_correctness</td><td>0.0</td><td>0.2</td></tr>
              <tr><td>citation_presence</td><td>0.0</td><td>0.1</td></tr>
              <tr><td>decision_label_consistency</td><td>0.0</td><td>0.1</td></tr>
              <tr><td>avg explanation length (words)</td><td>0.0</td><td>34.2</td></tr>
            </tbody>
          </table>
        </div>
        <div className="note">
          Partial format compliance at 300 iters. Generation bug (wrong base model) was fixed.
          Not production-grade; more training would improve compliance.
        </div>
      </div>

      <div className="section">
        <h3>What Must Not Be Claimed</h3>
        <div className="card">
          <ul style={{ paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: "var(--text-muted)" }}>
            <li>ONNX is faster than native transformers on this Mac (it is not — 55 vs 62 ex/s)</li>
            <li>Robust verifier improved retrieved performance (both metrics regressed)</li>
            <li>Relevance gate improved macro-F1 (NEI FPR improved, macro-F1 regressed)</li>
            <li>Phi-3 QLoRA or DPO adapter was trained (CUDA not available)</li>
            <li>MLX explanation adapter achieves production-quality output (format correctness = 0.2)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
