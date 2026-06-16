import { useEffect, useState } from "react";
import { fetchMetadata } from "./api";
import type { MetadataResponse, Tab } from "./types";
import VerifyClaim from "./components/VerifyClaim";
import EvidenceExplorer from "./components/EvidenceExplorer";
import Explanation from "./components/Explanation";
import FailureAnalysis from "./components/FailureAnalysis";
import ResearchMetrics from "./components/ResearchMetrics";
import TrainingArtifacts from "./components/TrainingArtifacts";

type NavItem = { id: Tab; label: string; group: "workspace" | "research"; icon: JSX.Element };

const NAV: NavItem[] = [
  {
    id: "verify", label: "Verify Claim", group: "workspace",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8 1.5L2.5 4v4.5C2.5 11.5 5 13.5 8 14.5c3-1 5.5-3 5.5-6V4L8 1.5z"/><path d="M5.5 8.5l1.8 1.8 3-3.6"/></svg>,
  },
  {
    id: "evidence", label: "Evidence", group: "workspace",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><circle cx="7" cy="7" r="4.5"/><path d="M14.5 14.5l-3.2-3.2"/></svg>,
  },
  {
    id: "explanation", label: "Explanation", group: "workspace",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M14 10.5a2 2 0 01-2 2H5l-3 3V3a2 2 0 012-2h8a2 2 0 012 2v7.5z"/></svg>,
  },
  {
    id: "failure", label: "Failure Analysis", group: "research",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8 1.5L1.5 13.5h13L8 1.5z"/><path d="M8 6v4M8 11.5v.5"/></svg>,
  },
  {
    id: "metrics", label: "Research Metrics", group: "research",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M2.5 13.5V9m3.5 4.5V5.5m3.5 8V3m3.5 10.5V7"/></svg>,
  },
  {
    id: "training", label: "Training", group: "research",
    icon: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8 1.5L1.5 5v6L8 14.5l6.5-3.5V5L8 1.5z"/><path d="M8 1.5v13M1.5 5l6.5 3.5L14.5 5"/></svg>,
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("verify");
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "ok" | "err">("loading");

  useEffect(() => {
    fetchMetadata()
      .then((m) => { setMetadata(m); setApiStatus("ok"); })
      .catch(() => setApiStatus("err"));
  }, []);

  const workspace = NAV.filter((n) => n.group === "workspace");
  const research  = NAV.filter((n) => n.group === "research");

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8l3.5 3.5L13 4"/>
            </svg>
          </div>
          <span className="brand-name">Veritas</span>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-group-label">Workspace</div>
          {workspace.map((item) => (
            <button key={item.id} className={`nav-item ${tab === item.id ? "active" : ""}`} onClick={() => setTab(item.id)}>
              {item.icon}{item.label}
            </button>
          ))}

          <div className="nav-group-label">Research</div>
          {research.map((item) => (
            <button key={item.id} className={`nav-item ${tab === item.id ? "active" : ""}`} onClick={() => setTab(item.id)}>
              {item.icon}{item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="api-status">
            <div className={`status-dot ${apiStatus === "ok" ? "green" : apiStatus === "err" ? "red" : "pulse"}`} />
            <span>
              {apiStatus === "loading" ? "Connecting…" : apiStatus === "ok" ? "API online" : "API offline — demo mode"}
            </span>
          </div>
        </div>
      </aside>

      <div className="content-area">
        <main key={tab} className="page">
          {tab === "verify"      && <VerifyClaim apiStatus={apiStatus} />}
          {tab === "evidence"    && <EvidenceExplorer />}
          {tab === "explanation" && <Explanation />}
          {tab === "failure"     && <FailureAnalysis />}
          {tab === "metrics"     && <ResearchMetrics />}
          {tab === "training"    && <TrainingArtifacts />}
        </main>
      </div>
    </div>
  );
}
