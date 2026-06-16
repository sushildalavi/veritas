import { Fragment, useRef, useState } from "react";
import { runPipeline } from "../api";
import type { EvidenceItem, PipelineResponse } from "../types";

type Phase = "idle" | "retrieving" | "verifying" | "explaining" | "done" | "error";

const DEMO_CLAIM = "The Apollo 11 mission landed humans on the Moon in 1969.";

const DEMO_RESULT: PipelineResponse = {
  request_id: "demo-apollo11",
  claim: DEMO_CLAIM,
  verdict: "SUPPORTED",
  confidence: 0.842,
  evidence: [
    {
      doc_id: "demo-nasa-001",
      source: "NASA Technical Reports",
      title: "Apollo 11 Mission Overview",
      text: "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Armstrong became the first human to step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC.",
      score: 0.8821,
      citation_id: 1,
      relation: "SUPPORTS",
    },
    {
      doc_id: "demo-britannica-001",
      source: "Encyclopædia Britannica",
      title: "Apollo 11 — Lunar Landing Mission",
      text: "Apollo 11, U.S. spaceflight during which Commander Neil Armstrong and lunar module pilot Edwin ('Buzz') Aldrin Jr. on July 20, 1969, became the first humans to land on the Moon. Armstrong, as he stepped out of the lunar module, said 'That's one small step for man, one giant leap for mankind.'",
      score: 0.8432,
      citation_id: 2,
      relation: "SUPPORTS",
    },
    {
      doc_id: "demo-wiki-001",
      source: "Wikipedia",
      title: "Apollo 11",
      text: "The mission's lunar module, Eagle, landed in the Sea of Tranquility (Mare Tranquillitatis) on July 20, 1969, at 20:17 UTC. Armstrong and Aldrin were on the lunar surface for two hours and 31 minutes and collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth.",
      score: 0.7914,
      citation_id: 3,
      relation: "SUPPORTS",
    },
  ],
  explanation:
    "The Apollo 11 mission did land humans on the Moon in 1969 [1]. Commander Neil Armstrong and lunar module pilot Buzz Aldrin became the first humans on the lunar surface on July 20, 1969 [2]. The lunar module Eagle landed in the Sea of Tranquility and the crew spent approximately two hours and 31 minutes on the surface [3].",
  citations: ["[1]", "[2]", "[3]"],
  citation_valid: true,
  backend_used: "transformer_verifier_clean",
  retrieval_backend: "bm25",
  explanation_mode: "demo_fixture",
  latency: {
    retrieval_ms: 12.3,
    verification_ms: 18.7,
    explanation_ms: 142.5,
    total_ms: 173.5,
  },
};

function verdictClass(v: string) {
  if (v === "SUPPORTED") return "supported";
  if (v === "REFUTED") return "refuted";
  return "nei";
}

function relationClass(r?: string) {
  if (r === "SUPPORTS") return "supports";
  if (r === "REFUTES") return "refutes";
  return "neutral";
}

function parseExplanation(text: string) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) return <sup key={i} className="cite-marker">{m[1]}</sup>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

function evidenceQuality(ev: EvidenceItem[]): string {
  const n = ev.filter((e) => e.relation === "SUPPORTS").length;
  return `${n} / ${ev.length} supporting`;
}

const PHASES: { id: Phase; label: string }[] = [
  { id: "retrieving", label: "Retrieve" },
  { id: "verifying",  label: "Verify" },
  { id: "explaining", label: "Explain" },
];
const PHASE_ORDER = ["retrieving", "verifying", "explaining", "done"];

function phaseStatus(current: Phase, step: string) {
  const ci = PHASE_ORDER.indexOf(current);
  const si = PHASE_ORDER.indexOf(step);
  if (ci > si) return "done";
  if (ci === si) return "active";
  return "waiting";
}

interface Props { apiStatus: "loading" | "ok" | "err"; }

export default function VerifyClaim({ apiStatus }: Props) {
  const [claim, setClaim] = useState(DEMO_CLAIM);
  const [topK, setTopK] = useState(5);
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const t1 = useRef<ReturnType<typeof setTimeout>>();
  const t2 = useRef<ReturnType<typeof setTimeout>>();

  async function handleRun() {
    if (!claim.trim()) return;
    setResult(null);
    setError(null);
    setIsDemo(false);
    setPhase("retrieving");

    t1.current = setTimeout(() => setPhase("verifying"), 450);
    t2.current = setTimeout(() => setPhase("explaining"), 950);

    if (apiStatus !== "ok") {
      // fall back to demo fixture
      clearTimeout(t1.current);
      clearTimeout(t2.current);
      await new Promise((r) => setTimeout(r, 1400));
      setResult(DEMO_RESULT);
      setIsDemo(true);
      setPhase("done");
      return;
    }

    try {
      const data = await runPipeline(claim.trim(), topK);
      clearTimeout(t1.current);
      clearTimeout(t2.current);
      setResult(data);
      setPhase("done");
    } catch (e: unknown) {
      clearTimeout(t1.current);
      clearTimeout(t2.current);
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }

  const loading = phase !== "idle" && phase !== "done" && phase !== "error";

  return (
    <div>
      <div className="page-header">
        <div className="hero-title">Verify Claim</div>
        <div className="page-desc">
          Enter any factual claim. Veritas retrieves evidence via BM25, classifies each passage with
          DistilRoBERTa NLI, aggregates to a verdict, and generates a citation-backed explanation.
        </div>
      </div>

      {/* ─── Command bar ─── */}
      <div className={`command-bar ${loading ? "loading" : ""}`}>
        <textarea
          className="command-input"
          placeholder="Enter a claim to verify…"
          value={claim}
          rows={2}
          onChange={(e) => setClaim(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleRun(); }}
          disabled={loading}
        />
        <div className="command-footer">
          <div className="command-hints">
            <span className="kbd">⌘ Enter</span>
            <span style={{ fontSize: "11px", color: "var(--text-dim)" }}>to run</span>
          </div>
          <div className="command-actions">
            <div className="topk-row">
              <label>Top-K</label>
              <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} disabled={loading}>
                {[3, 5, 10].map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </div>
            <button className="btn" onClick={handleRun} disabled={loading || !claim.trim()}>
              {loading ? <><span className="spinner" />Running…</> : "Run pipeline"}
            </button>
          </div>
        </div>
      </div>

      {/* ─── State progress ─── */}
      {loading && (
        <div className="verify-progress">
          {PHASES.map((p, i) => {
            const s = phaseStatus(phase, p.id);
            return (
              <Fragment key={p.id}>
                <div className="vp-step">
                  <div className={`vp-dot ${s}`} />
                  <div className={`vp-label ${s}`}>{p.label}</div>
                </div>
                {i < PHASES.length - 1 && (
                  <div className={`vp-line ${phaseStatus(phase, PHASES[i + 1].id) === "waiting" ? "active" : "done"}`} />
                )}
              </Fragment>
            );
          })}
        </div>
      )}

      {/* ─── Error ─── */}
      {phase === "error" && error && (
        <div className="notice notice-error">{error}</div>
      )}

      {/* ─── Results ─── */}
      {result && phase === "done" && (
        <>
          {/* Verdict row */}
          <div className="verdict-row">
            <div className="verdict-left">
              <div className="verdict-sublabel">Verdict</div>
              <div className={`verdict-badge-lg ${verdictClass(result.verdict)}`}>
                {result.verdict === "SUPPORTED" && "✓ "}
                {result.verdict === "REFUTED"   && "✗ "}
                {result.verdict === "NOT ENOUGH INFO" && "? "}
                {result.verdict}
              </div>
            </div>

            <div className="status-mini">
              <div className="status-mini-item">
                <div className="smi-label">Confidence</div>
                <div className={`smi-value ${result.confidence >= 0.7 ? "ok" : result.confidence >= 0.5 ? "warn" : "error"}`}>
                  {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className="status-mini-item">
                <div className="smi-label">Evidence quality</div>
                <div className="smi-value ok">{evidenceQuality(result.evidence)}</div>
              </div>
              <div className="status-mini-item">
                <div className="smi-label">Citations</div>
                <div className={`smi-value ${result.citation_valid ? "ok" : "warn"}`}>
                  {result.citation_valid ? "valid" : "none"}
                </div>
              </div>
              {isDemo && (
                <div className="status-mini-item">
                  <div className="smi-label">Source</div>
                  <span className="demo-badge">demo fixture</span>
                </div>
              )}
            </div>
          </div>

          {/* Explanation */}
          <div className="section">
            <div className="section-label">Explanation</div>
            <div className="answer-box">
              <div className="answer-text">
                {parseExplanation(result.explanation || "No explanation generated.")}
              </div>
              {result.evidence.length > 0 && (
                <div className="citation-chips">
                  {result.evidence.map((ev, i) => (
                    <div key={ev.doc_id + i} className="citation-chip">
                      <span className="chip-num">[{ev.citation_id ?? i + 1}]</span>
                      <span>{ev.source ?? ev.title ?? ev.doc_id}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Evidence cards */}
          <div className="section">
            <div className="section-label">Retrieved evidence ({result.evidence.length} passages)</div>
            <div className="evidence-cards">
              {result.evidence.map((ev, i) => (
                <div key={ev.doc_id + i} className="ev-card">
                  <div className="ev-card-head">
                    <div className="ev-rank">#{ev.citation_id ?? i + 1}</div>
                    <div className={`ev-relation ${relationClass(ev.relation)}`}>
                      {ev.relation ?? "NEUTRAL"}
                    </div>
                    {ev.score != null && (
                      <div className="ev-score">score {ev.score.toFixed(4)}</div>
                    )}
                  </div>
                  <div className="ev-card-body">
                    {ev.source && <div className="ev-source">{ev.source}</div>}
                    {ev.title  && <div className="ev-title">{ev.title}</div>}
                    <div className="ev-text">{ev.text}</div>
                    <div className="ev-footer">
                      <div className="ev-docid">{ev.doc_id}</div>
                      <div className="ev-cite-id">[{ev.citation_id ?? i + 1}]</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Latency */}
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "24px" }}>
            {[
              ["retrieval",     result.latency.retrieval_ms],
              ["verification",  result.latency.verification_ms],
              ["explanation",   result.latency.explanation_ms],
              ["total",         result.latency.total_ms],
            ].map(([k, v]) => (
              <div key={k as string} style={{
                fontSize: "11px", fontFamily: "var(--mono)",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border)", borderRadius: "5px",
                padding: "3px 10px", color: "var(--text-dim)",
              }}>
                {k as string}<span style={{ color: "var(--text-muted)", marginLeft: "5px" }}>{(v as number).toFixed(1)} ms</span>
              </div>
            ))}
          </div>

          {/* Uncertainty panel */}
          <div className="uncertainty-panel">
            <div className="up-title">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M8 1.5L1.5 13.5h13L8 1.5z"/><path d="M8 6v4M8 11.5v.5"/>
              </svg>
              Evidence retrieval is the bottleneck
            </div>
            <div className="up-item"><div className="up-dot"/><span>Retrieved macro-F1 is 0.3887 vs oracle 0.6728 — a gap of 0.2841. BM25 recall@10 is 0.5334.</span></div>
            <div className="up-item"><div className="up-dot"/><span>Production verifier is unchanged (DistilRoBERTa NLI). Robust retrain regressed to 0.2829.</span></div>
            <div className="up-item"><div className="up-dot"/><span>This is a local research prototype — not a production fact checker or SOTA system.</span></div>
            {isDemo && (
              <div className="up-item"><div className="up-dot"/><span>Results shown are demo fixtures (backend offline). Run <code className="code">make api</code> for live verification.</span></div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
