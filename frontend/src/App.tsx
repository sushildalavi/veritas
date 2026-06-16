import { useEffect, useState } from "react";
import { fetchMetadata } from "./api";
import type { MetadataResponse, Tab } from "./types";
import Overview from "./components/Overview";
import VerifyClaim from "./components/VerifyClaim";
import EvidenceExplorer from "./components/EvidenceExplorer";
import TrainingArtifacts from "./components/TrainingArtifacts";
import ResearchResults from "./components/ResearchResults";

const NAV_ITEMS: { id: Tab; label: string; icon: JSX.Element }[] = [
  {
    id: "overview",
    label: "Overview",
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M1.5 6.75L8 1.5l6.5 5.25V13.5a.5.5 0 01-.5.5H10V9.5H6V14H2a.5.5 0 01-.5-.5V6.75z"/>
      </svg>
    ),
  },
  {
    id: "verify",
    label: "Verify Claim",
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 1.5L2.5 4v4.5C2.5 11.5 5 13.5 8 14.5c3-1 5.5-3 5.5-6V4L8 1.5z"/>
        <path d="M5.5 8.5l1.8 1.8 3-3.6"/>
      </svg>
    ),
  },
  {
    id: "evidence",
    label: "Evidence",
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <circle cx="7" cy="7" r="4.5"/>
        <path d="M14.5 14.5l-3.2-3.2"/>
      </svg>
    ),
  },
  {
    id: "training",
    label: "Training",
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 1.5L1.5 5v6L8 14.5l6.5-3.5V5L8 1.5z"/>
        <path d="M8 1.5v13M1.5 5l6.5 3.5L14.5 5"/>
      </svg>
    ),
  },
  {
    id: "results",
    label: "Results",
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <path d="M2.5 13.5V9m3.5 4.5V5.5m3.5 8V3m3.5 10.5V7"/>
      </svg>
    ),
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "ok" | "err">("loading");

  useEffect(() => {
    fetchMetadata()
      .then((m) => { setMetadata(m); setApiStatus("ok"); })
      .catch((e: unknown) => {
        setApiError(e instanceof Error ? e.message : "API unreachable");
        setApiStatus("err");
      });
  }, []);

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8l3.5 3.5L13 4"/>
            </svg>
          </div>
          <span className="brand-name">Veritas</span>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">Navigation</div>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${tab === item.id ? "active" : ""}`}
              onClick={() => setTab(item.id)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="api-status">
            <div className={`status-dot ${apiStatus === "ok" ? "green" : apiStatus === "err" ? "red" : "pulse"}`} />
            <span>
              {apiStatus === "loading" ? "Connecting…" : apiStatus === "ok" ? "API online" : "API offline"}
            </span>
          </div>
        </div>
      </aside>

      <div className="content-area">
        <main className="page">
          {tab === "overview" && <Overview metadata={metadata} apiError={apiStatus === "err" ? apiError : null} />}
          {tab === "verify" && <VerifyClaim />}
          {tab === "evidence" && <EvidenceExplorer />}
          {tab === "training" && <TrainingArtifacts />}
          {tab === "results" && <ResearchResults />}
        </main>
      </div>
    </div>
  );
}
