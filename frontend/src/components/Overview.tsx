import type { MetadataResponse } from "../types";
import { CountUp } from "./CountUp";

interface Props {
  metadata: MetadataResponse | null;
  apiError: string | null;
}

const F1_BARS = [
  { label: "Oracle macro-F1",       value: 0.6728, color: "#3b82f6", delay: "0.15s" },
  { label: "Retrieved macro-F1",    value: 0.3887, color: "#f59e0b", delay: "0.28s" },
  { label: "Oracle → Ret. gap",     value: 0.2841, color: "#ef4444", delay: "0.41s" },
  { label: "Recall@10 (BM25)",      value: 0.5334, color: "#22c55e", delay: "0.54s" },
];

export default function Overview({ metadata, apiError }: Props) {
  return (
    <div>
      <div className="page-header">
        <div className="hero-title">Veritas</div>
        <div className="page-desc">
          Evidence-grounded fact verification — BM25 retrieval, DistilRoBERTa NLI,
          oracle-vs-retrieved ablations, MLX LoRA training, and a FastAPI research backend.
          All metrics measured on 650 examples (FEVER + SciFact).
        </div>
      </div>

      {apiError && (
        <div className="notice notice-error" style={{ marginBottom: "24px" }}>
          Backend unavailable — {apiError}. Run <code className="code">make api</code> to start.
        </div>
      )}

      <div className="section">
        <div className="section-label">Key Measured Results</div>
        <div className="metric-grid">
          <div className="metric-card accent">
            <div className="metric-label">Oracle macro-F1</div>
            <div className="metric-value">
              <CountUp to={0.6728} decimals={4} duration={1300} />
            </div>
            <div className="metric-sub">per_passage_max · gold evidence</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Retrieved macro-F1</div>
            <div className="metric-value">
              <CountUp to={0.3887} decimals={4} duration={1300} delay={80} />
            </div>
            <div className="metric-sub">per_passage_max · BM25 retrieved</div>
          </div>
          <div className="metric-card negative">
            <div className="metric-label">Oracle → Retrieved gap</div>
            <div className="metric-value">
              <CountUp to={0.2841} decimals={4} duration={1300} delay={160} />
            </div>
            <div className="metric-sub">retrieval is primary bottleneck</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Recall@10</div>
            <div className="metric-value">
              <CountUp to={0.5334} decimals={4} duration={1300} delay={240} />
            </div>
            <div className="metric-sub">BM25 default profile</div>
          </div>
          {metadata && (
            <div className="metric-card" style={{ borderColor: "var(--green-border)" }}>
              <div className="metric-label">API status</div>
              <div className="metric-value" style={{ fontSize: "22px", letterSpacing: "-0.5px", color: "var(--green)" }}>
                Online
              </div>
              <div className="metric-sub">{metadata.retrieval_profile} · v{metadata.version}</div>
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-label">Performance Gap</div>
        <div className="card">
          <div className="bar-chart">
            {F1_BARS.map((b) => (
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
      </div>

      <div className="section">
        <div className="section-label">Pipeline</div>
        <div className="card" style={{ padding: "16px 20px" }}>
          <div className="pipeline">
            <div className="pipeline-step">
              <div className="pipeline-step-name">Claim</div>
              <div className="pipeline-step-detail">input text</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-name">BM25 Retrieval</div>
              <div className="pipeline-step-detail">recall@10 0.5334</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-name">NLI Verifier</div>
              <div className="pipeline-step-detail">oracle F1 0.6728</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-name">Aggregation</div>
              <div className="pipeline-step-detail">per_passage_max</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-name">Explanation</div>
              <div className="pipeline-step-detail">MLX LoRA · rule</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-name">FastAPI</div>
              <div className="pipeline-step-detail">8 endpoints</div>
            </div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-label">Negative Results</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Outcome</th>
                <th>Retrieved macro-F1</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Robust verifier retrain</td>
                <td><span className="badge badge-red">Regression</span></td>
                <td>0.3887 → 0.2829 (−0.1058)</td>
                <td>NEI collapse; production checkpoint unchanged</td>
              </tr>
              <tr>
                <td>Relevance gate (threshold 0.5)</td>
                <td><span className="badge badge-red">Regression</span></td>
                <td>0.3887 → 0.3557 (−0.0330)</td>
                <td>NEI FPR improved 0.71→0.29; gate disabled</td>
              </tr>
              <tr>
                <td>Hybrid sentence-transformer</td>
                <td><span className="badge badge-amber">Mixed</span></td>
                <td>0.3887 → 0.3776 (−0.0111)</td>
                <td>Better recall@10 (0.53→0.57) but worse F1</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <div className="section-label">Quick Start</div>
        <div className="card">
          <pre style={{ fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "2.1", color: "var(--text-muted)" }}>
{`make api        # start the FastAPI backend on :8000
make frontend   # start the Vite dev server on :5173`}
          </pre>
        </div>
      </div>

      <div className="notice notice-neutral" style={{ marginTop: "4px" }}>
        Research prototype · not production-deployed · retrieved macro-F1 0.3887 · all metrics measured on 650-example test set
      </div>
    </div>
  );
}
