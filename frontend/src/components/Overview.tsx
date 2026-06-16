import type { MetadataResponse } from "../types";

interface Props {
  metadata: MetadataResponse | null;
  apiError: string | null;
}

export default function Overview({ metadata, apiError }: Props) {
  return (
    <div>
      <div className="page-header">
        <div className="page-title">Veritas</div>
        <div className="page-desc">
          Evidence-grounded fact verification system — BM25 retrieval, DistilRoBERTa NLI verification,
          oracle-vs-retrieved evaluation, MLX LoRA training, and a FastAPI research backend.
          All metrics are measured on a 650-example test set (FEVER + SciFact).
        </div>
      </div>

      {apiError && (
        <div className="notice notice-error" style={{ marginBottom: "24px" }}>
          Backend unavailable — {apiError}. Start the API with <code className="code">make api</code>.
        </div>
      )}

      <div className="section">
        <div className="section-label">Key Measured Results</div>
        <div className="metric-grid">
          <div className="metric-card accent">
            <div className="metric-label">Oracle macro-F1</div>
            <div className="metric-value">0.6728</div>
            <div className="metric-sub">per_passage_max · gold evidence</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Retrieved macro-F1</div>
            <div className="metric-value">0.3887</div>
            <div className="metric-sub">per_passage_max · BM25 retrieved</div>
          </div>
          <div className="metric-card negative">
            <div className="metric-label">Oracle → Retrieved gap</div>
            <div className="metric-value">0.2841</div>
            <div className="metric-sub">retrieval is primary bottleneck</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Recall@10</div>
            <div className="metric-value">0.5334</div>
            <div className="metric-sub">BM25 default profile</div>
          </div>
          {metadata && (
            <div className="metric-card">
              <div className="metric-label">API status</div>
              <div className="metric-value" style={{ fontSize: "20px", color: "var(--green)" }}>Online</div>
              <div className="metric-sub">{metadata.retrieval_profile} profile</div>
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-label">Pipeline</div>
        <div className="card">
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
              <div className="pipeline-step-detail">MLX LoRA / rule</div>
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
                <td>NEI FPR improved (0.71→0.29); gate disabled by default</td>
              </tr>
              <tr>
                <td>Hybrid sentence-transformer</td>
                <td><span className="badge badge-amber">Mixed</span></td>
                <td>0.3887 → 0.3776 (−0.0111)</td>
                <td>Better recall@10 (0.53→0.57) but worse verifier F1</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <div className="section-label">Quick Start</div>
        <div className="card">
          <pre style={{ fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "2", color: "var(--text-muted)" }}>
{`# 1. Start the backend
make api

# 2. Start the frontend (separate terminal)
make frontend

# 3. Open http://localhost:5173`}
          </pre>
        </div>
      </div>

      <div className="notice notice-neutral">
        Research prototype · not production-deployed · retrieved macro-F1 0.3887 · all metrics measured on 650-example test set
      </div>
    </div>
  );
}
