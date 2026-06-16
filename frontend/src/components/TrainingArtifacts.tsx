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
        <h3>Production Status</h3>
        <div className="status-grid">
          <div className="status-item">
            <div className="dot ok" />
            <div>
              <div style={{ fontWeight: 600 }}>Production verifier</div>
              <div style={{ color: "var(--text-muted)", fontSize: "11px" }}>transformer_verifier_clean</div>
            </div>
          </div>
          <div className="status-item">
            <div className="dot warn" />
            <div>
              <div style={{ fontWeight: 600 }}>Relevance gate</div>
              <div style={{ color: "var(--text-muted)", fontSize: "11px" }}>disabled by default</div>
            </div>
          </div>
          <div className="status-item">
            <div className="dot warn" />
            <div>
              <div style={{ fontWeight: 600 }}>MLX LoRA explanation</div>
              <div style={{ color: "var(--text-muted)", fontSize: "11px" }}>trained; partial format compliance</div>
            </div>
          </div>
          <div className="status-item">
            <div className="dot err" />
            <div>
              <div style={{ fontWeight: 600 }}>Phi-3 QLoRA / DPO</div>
              <div style={{ color: "var(--text-muted)", fontSize: "11px" }}>CUDA path only; not trained locally</div>
            </div>
          </div>
        </div>
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
                <td>500 iters; citation_presence 0.72; partial format compliance</td>
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
        <h3>MLX LoRA — Explanation SFT (measured at 500 iters)</h3>
        <p style={{ fontSize: "12px", marginBottom: "8px" }}>
          Base model: Qwen2.5-1.5B-Instruct-4bit · 256 training examples · Apple Silicon
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Metric</th><th>Base model</th><th>Adapter (300 iters)</th><th>Adapter (500 iters)</th></tr>
            </thead>
            <tbody>
              <tr><td>format_correctness</td><td>0.0</td><td>0.20</td><td>0.28</td></tr>
              <tr className="bold-row"><td>citation_presence</td><td>0.0</td><td>0.10</td><td>0.72</td></tr>
              <tr><td>decision_label_consistency</td><td>0.0</td><td>0.10</td><td>0.24</td></tr>
              <tr><td>avg explanation length (words)</td><td>0.0</td><td>34.2</td><td>33.2</td></tr>
            </tbody>
          </table>
        </div>
        <div className="note">
          Generation bug (base_model key mismatch) was fixed and adapter retrained.
          Citation presence improved from 0.10 → 0.72. Partial format compliance.
          <strong> Not production-grade</strong> — small-sample training (256 examples, 500 iters).
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
