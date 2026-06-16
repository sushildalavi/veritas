import { Fragment, useMemo, useState } from "react";
import { checkClaim } from "../api";
import type { HealthResponse, MetadataResponse, PipelineResponse } from "../types";

const EXAMPLES = [
  "The Pacific Ocean is the largest ocean.",
  "The Pacific Ocean is the largest ocean on Earth.",
  "Water freezes at 0 degrees Celsius at sea level.",
  "Paris is the capital city of France.",
];

const EVIDENCE_DEPTHS = [3, 5, 8] as const;

type ServiceState = {
  ready: boolean;
  loading: boolean;
  health: HealthResponse | null;
  metadata: MetadataResponse | null;
  error: string | null;
};

function verdictLabel(v: string) {
  if (v === "SUPPORTED") return "Supported";
  if (v === "REFUTED") return "Refuted";
  return "Inconclusive";
}

function verdictIcon(v: string) {
  if (v === "SUPPORTED") return "✓";
  if (v === "REFUTED") return "✕";
  return "·";
}

function verdictClass(v: string) {
  if (v === "SUPPORTED") return "verdict-supported";
  if (v === "REFUTED") return "verdict-refuted";
  return "verdict-neutral";
}

function confidenceLabel(value: number) {
  if (value >= 0.7) return "High confidence";
  if (value >= 0.55) return "Borderline";
  return "Review recommended";
}

function confidenceTone(value: number) {
  if (value >= 0.7) return "confidence-high";
  if (value >= 0.55) return "confidence-mid";
  return "confidence-low";
}

function relationLabel(r?: string) {
  if (r === "SUPPORTS") return "Supports";
  if (r === "REFUTES") return "Refutes";
  return "Neutral";
}

function relationClass(r?: string) {
  if (r === "SUPPORTS") return "relation-supports";
  if (r === "REFUTES") return "relation-refutes";
  return "relation-neutral";
}

function parseExplanation(text: string) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const citation = part.match(/^\[(\d+)\]$/);
    if (citation) {
      return (
        <sup key={index} className="inline-citation">
          {citation[1]}
        </sup>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function formatMs(value: number) {
  if (value < 10) return `${value.toFixed(1)} ms`;
  if (value < 100) return `${value.toFixed(0)} ms`;
  return `${Math.round(value)} ms`;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

interface Props {
  service: ServiceState;
}

export default function VerifyClaim({ service }: Props) {
  const [claim, setClaim] = useState("");
  const [evidenceDepth, setEvidenceDepth] = useState<(typeof EVIDENCE_DEPTHS)[number]>(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const maxClaimLength = service.health?.max_claim_length ?? 1000;
  const remaining = maxClaimLength - claim.length;
  const overLimit = remaining < 0;
  const canSubmit = claim.trim().length > 0 && !loading && !overLimit && service.ready;
  const topEvidenceScore = result?.evidence[0]?.score ?? null;
  const groundingState =
    topEvidenceScore === null
      ? null
      : topEvidenceScore >= 3
        ? {
            label: "Strong grounding",
            tone: "grounding-strong",
            note: "Top evidence closely matches the claim.",
          }
        : topEvidenceScore >= 1.5
          ? {
              label: "Moderate grounding",
              tone: "grounding-moderate",
              note: "The answer is usable, but still worth skimming.",
            }
          : {
              label: "Weak grounding",
              tone: "grounding-weak",
              note: "The retrieved evidence is thin or off-target.",
            };

  const serviceBadges = useMemo(() => {
    const badges = [
      service.health?.verifier_backend && `Verifier: ${service.health.verifier_backend}`,
      service.health?.retrieval_backend && `Retrieval: ${service.health.retrieval_backend}`,
      service.health?.model_name && `Model: ${service.health.model_name}`,
    ].filter(Boolean) as string[];

    if (service.metadata?.retrieval_profile) {
      badges.push(`Profile: ${service.metadata.retrieval_profile}`);
    }

    return badges;
  }, [service.health, service.metadata]);

  async function handleVerify() {
    const text = claim.trim();
    if (!text || loading || overLimit) return;

    setLoading(true);
    setResult(null);
    setError(null);

    if (!service.ready) {
      setError("Backend unavailable. Start the FastAPI service and try again.");
      setLoading(false);
      return;
    }

    try {
      const data = await checkClaim(text, evidenceDepth);
      setResult(data);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }

  function loadExample(text: string) {
    setClaim(text);
    setError(null);
    setResult(null);
  }

  function resetWorkspace() {
    setClaim("");
    setError(null);
    setResult(null);
  }

  const verdict = result ? verdictLabel(result.verdict) : "Awaiting claim";

  return (
    <div className="workspace">
      <section className="hero-card">
        <div className="hero-copy">
          <div className="eyebrow">Claim review workflow</div>
          <h1>Verify a claim with grounded evidence, not guesswork.</h1>
          <p>
            Veritas retrieves evidence, scores the claim, and drafts a citation-grounded
            explanation so review teams can separate strong claims from ones that need
            another look.
          </p>
        </div>

        <div className="hero-panel">
          <div className="panel-title">System validation</div>
          <div className="metric-grid">
            <div className="metric-card">
              <span className="metric-label">Verifier F1</span>
              <strong>{service.health?.verifier_macro_f1 ? service.health.verifier_macro_f1.toFixed(4) : "n/a"}</strong>
              <span className="metric-note">Measured on the bundled checkpoint</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Retrieval recall@10</span>
              <strong>{service.metadata ? service.metadata.retrieval_recall_at_10.toFixed(4) : "n/a"}</strong>
              <span className="metric-note">Validation of the retrieval layer</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Oracle gap</span>
              <strong>{service.metadata ? service.metadata.oracle_retrieved_gap.toFixed(4) : "n/a"}</strong>
              <span className="metric-note">What gets harder once evidence is noisy</span>
            </div>
          </div>
        </div>
      </section>

      <section className="status-row" aria-label="Service details">
        {service.loading ? (
          <div className="status-message">Checking service status and metadata…</div>
        ) : service.error && !service.ready ? (
          <div className="status-message status-warning">
            Backend check failed. {service.error}
          </div>
        ) : (
          <>
            {serviceBadges.map((badge) => (
              <span key={badge} className="status-pill">
                {badge}
              </span>
            ))}
            {service.health?.fallback_used && (
              <span className="status-pill status-pill-warn">Verifier fallback active</span>
            )}
            {service.health?.retrieval_fallback_used && (
              <span className="status-pill status-pill-warn">Retrieval fallback active</span>
            )}
          </>
        )}
      </section>

      <section className="workspace-grid">
        <div className="card claim-card">
          <div className="card-header">
            <div>
              <div className="card-kicker">Input</div>
              <h2>Claim to verify</h2>
            </div>
            <button className="ghost-button" type="button" onClick={resetWorkspace}>
              Reset
            </button>
          </div>

          <label className="field-label" htmlFor="claim-input">
            Paste a single factual statement
          </label>
          <textarea
            id="claim-input"
            className="claim-input"
            placeholder="Example: Paris is the capital city of France."
            value={claim}
            rows={6}
            onChange={(event) => setClaim(event.target.value)}
            disabled={loading}
          />

          <div className="field-meta">
            <span className={overLimit ? "count count-over" : "count"}>
              {formatCount(claim.length)} / {formatCount(maxClaimLength)} chars
            </span>
            <span className="helper-text">
              Evidence depth controls how many passages reach the verifier.
            </span>
          </div>
          {overLimit && (
            <div className="inline-warning">
              Claim exceeds the API limit by {Math.abs(remaining)} characters.
            </div>
          )}

          <div className="depth-group" role="group" aria-label="Evidence depth">
            {EVIDENCE_DEPTHS.map((value) => (
              <button
                key={value}
                type="button"
                className={`depth-chip ${value === evidenceDepth ? "depth-chip-active" : ""}`}
                onClick={() => setEvidenceDepth(value)}
              >
                <span>{value} passages</span>
                <small>{value === 3 ? "Fast" : value === 5 ? "Balanced" : "Deeper"}</small>
              </button>
            ))}
          </div>

          <div className="example-section">
            <span className="example-label">Good starting points</span>
            <div className="example-grid">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="example-button"
                  onClick={() => loadExample(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <div className="actions">
            <div className="shortcut-note">
              {loading ? "Working…" : service.ready ? "Runs the live pipeline." : "Start the API to verify claims."}
            </div>
            <button className="primary-button" type="button" onClick={handleVerify} disabled={!canSubmit}>
              {loading ? "Verifying…" : "Verify claim"}
            </button>
          </div>
        </div>

        <div className="card result-card">
          <div className="card-header">
            <div>
              <div className="card-kicker">Output</div>
              <h2>Review result</h2>
            </div>
            <div className="result-status">{verdict}</div>
          </div>

          {error && (
            <div className="error-notice" role="alert">
              {error}
            </div>
          )}

          {!result ? (
            <div className="empty-state">
              <div className="empty-title">No claim has been verified yet.</div>
              <p>
                Submit a statement to retrieve evidence, score the claim, and review the generated explanation.
              </p>
            </div>
          ) : (
            <>
              <div className="verdict-panel">
                <div className={`verdict-badge ${verdictClass(result.verdict)}`}>
                  <span className="verdict-icon">{verdictIcon(result.verdict)}</span>
                  <div>
                    <div className="verdict-label">{verdictLabel(result.verdict)}</div>
                    <div className="verdict-subtitle">Verifier output for this claim</div>
                  </div>
                </div>

                <div className="confidence-block">
                  <div className="confidence-top">
                    <span>Model confidence</span>
                    <strong>{formatPercent(result.confidence)}</strong>
                  </div>
                  <div className="confidence-bar" aria-hidden="true">
                    <div className="confidence-fill" style={{ width: formatPercent(result.confidence) }} />
                  </div>
                  <div className={`confidence-note ${confidenceTone(result.confidence)}`}>
                    {confidenceLabel(result.confidence)}. Evidence quality still matters.
                  </div>
                </div>
              </div>

              <div className="result-meta">
                <span className="result-pill">Citation check: {result.citation_valid ? "passed" : "needs review"}</span>
                <span className="result-pill">Verifier: {result.backend_used}</span>
                <span className="result-pill">Retrieval: {result.retrieval_backend}</span>
                <span className="result-pill">
                  Top evidence: {topEvidenceScore === null ? "n/a" : topEvidenceScore.toFixed(2)}
                </span>
              </div>

              {groundingState && (
                <div className={`grounding-callout ${groundingState.tone}`}>
                  <div className="grounding-title">{groundingState.label}</div>
                  <div className="grounding-note">{groundingState.note}</div>
                </div>
              )}

              <div className="explanation-block">
                <div className="section-label">Explanation</div>
                <p>{parseExplanation(result.explanation)}</p>
              </div>

              <div className="latency-grid">
                <div>
                  <span>Retrieval</span>
                  <strong>{formatMs(result.latency.retrieval_ms)}</strong>
                </div>
                <div>
                  <span>Verification</span>
                  <strong>{formatMs(result.latency.verification_ms)}</strong>
                </div>
                <div>
                  <span>Explanation</span>
                  <strong>{formatMs(result.latency.explanation_ms)}</strong>
                </div>
                <div>
                  <span>Total</span>
                  <strong>{formatMs(result.latency.total_ms)}</strong>
                </div>
              </div>

              <div className="section-header">
                <div className="section-label">Evidence</div>
                <div className="section-note">{result.evidence.length} passages reviewed</div>
              </div>

              <div className="evidence-list">
                {result.evidence.map((item, index) => (
                  <article key={item.doc_id + index} className="evidence-card">
                    <div className="evidence-top">
                      <div className="evidence-source">
                        <span className="evidence-index">[{item.citation_id ?? index + 1}]</span>
                        <span>{item.source ?? item.title ?? item.doc_id}</span>
                      </div>
                      <span className={`relation-pill ${relationClass(item.relation)}`}>
                        {relationLabel(item.relation)}
                      </span>
                    </div>
                    {item.title && <div className="evidence-title">{item.title}</div>}
                    <p className="evidence-text">{item.text}</p>
                    {typeof item.score === "number" && (
                      <div className="evidence-score">
                        <span>Alignment score</span>
                        <strong>{item.score.toFixed(2)}</strong>
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
