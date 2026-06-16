import type { MetadataResponse } from "../types";

interface Props {
  metadata: MetadataResponse | null;
  apiError: string | null;
}

export default function Overview({ metadata, apiError }: Props) {
  return (
    <div>
      <div className="section">
        <h1>Veritas</h1>
        <p style={{ fontSize: "15px", marginBottom: "10px" }}>
          Evidence-Grounded Fact Verification &amp; Training Platform
        </p>
        <p>
          A failure-aware fact-verification system with BM25 retrieval, DistilRoBERTa NLI
          verification, oracle-vs-retrieved evaluation, training artifacts, inference
          benchmarks, and a local research dashboard.
        </p>
        <div className="note" style={{ marginTop: "14px" }}>
          ⚠ Veritas is a research prototype — not a production-deployed fact checker.
          Retrieved macro-F1 is 0.3887; retrieval is the primary bottleneck.
        </div>
      </div>

      {apiError && (
        <div className="error-box">
          Backend unavailable: {apiError}. Start the API with <code className="code">make api</code>.
        </div>
      )}

      <div className="section">
        <h2>Key Measured Results</h2>
        <p style={{ marginBottom: "4px" }}>Full 650-example test set (FEVER + SciFact).</p>
        <div className="card-grid">
          <div className="metric-card highlight">
            <div className="label">Oracle macro-F1</div>
            <div className="value">0.6728</div>
            <div className="sub">per_passage_max, gold evidence</div>
          </div>
          <div className="metric-card">
            <div className="label">Retrieved macro-F1</div>
            <div className="value">0.3887</div>
            <div className="sub">per_passage_max, BM25 retrieved</div>
          </div>
          <div className="metric-card negative">
            <div className="label">Oracle→Retrieved Gap</div>
            <div className="value">0.2841</div>
            <div className="sub">retrieval is primary bottleneck</div>
          </div>
          <div className="metric-card">
            <div className="label">Retrieval recall@10</div>
            <div className="value">0.5334</div>
            <div className="sub">BM25 default profile</div>
          </div>
          {metadata && (
            <div className="metric-card">
              <div className="label">API status</div>
              <div className="value" style={{ fontSize: "18px", color: "var(--green)" }}>Online</div>
              <div className="sub">{metadata.retrieval_profile} profile</div>
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <h2>Architecture</h2>
        <div className="card">
          <pre style={{ fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "1.8", color: "var(--text-muted)", whiteSpace: "pre-wrap" }}>
{`Claim
  → BM25 retrieval (recall@10 = 0.5334)
  → DistilRoBERTa NLI verifier  (oracle macro-F1 = 0.6728)
  → per_passage_max aggregation (retrieved macro-F1 = 0.3887)
  → citation / explanation layer
  → FastAPI backend  (8 endpoints)
  → React research dashboard
  → full-set evaluation reports (650 examples)`}
          </pre>
        </div>
      </div>

      <div className="section">
        <h2>Negative Results</h2>
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
                <td><span className="neg-badge">Regression</span></td>
                <td>0.3887 → 0.2829 (−0.1058)</td>
                <td>NEI collapse; production checkpoint unchanged</td>
              </tr>
              <tr>
                <td>Relevance gate (threshold 0.5)</td>
                <td><span className="neg-badge">Regression</span></td>
                <td>0.3887 → 0.3557 (−0.0330)</td>
                <td>NEI FPR improved (0.71→0.29); gate disabled by default</td>
              </tr>
              <tr>
                <td>Hybrid sentence-transformer retrieval</td>
                <td><span className="neg-badge">Mixed</span></td>
                <td>0.3887 → 0.3776 (−0.0111)</td>
                <td>Better recall@10 (0.53→0.57) but worse verifier F1</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <h2>Quick Start</h2>
        <div className="card">
          <pre style={{ fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "2", color: "var(--text)" }}>
{`# 1. Start the backend
make api

# 2. Start the frontend (separate terminal)
make frontend

# 3. Open http://localhost:5173`}
          </pre>
        </div>
      </div>
    </div>
  );
}
