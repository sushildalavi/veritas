import { useEffect, useState } from "react";
import { fetchMetadata } from "./api";
import type { MetadataResponse, Tab } from "./types";
import Overview from "./components/Overview";
import VerifyClaim from "./components/VerifyClaim";
import EvidenceExplorer from "./components/EvidenceExplorer";
import TrainingArtifacts from "./components/TrainingArtifacts";
import ResearchResults from "./components/ResearchResults";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "verify", label: "Verify Claim" },
  { id: "evidence", label: "Evidence Explorer" },
  { id: "training", label: "Training Artifacts" },
  { id: "results", label: "Research Results" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "ok" | "err">("loading");

  useEffect(() => {
    fetchMetadata()
      .then((m) => {
        setMetadata(m);
        setApiStatus("ok");
      })
      .catch((e: unknown) => {
        setApiError(e instanceof Error ? e.message : "API unreachable");
        setApiStatus("err");
      });
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="topbar-title">Veritas</div>
          <div className="topbar-subtitle">Evidence-Grounded Fact Verification</div>
        </div>
        <div className={`topbar-badge ${apiStatus === "ok" ? "ok" : apiStatus === "err" ? "err" : ""}`}>
          {apiStatus === "loading" ? "connecting…" : apiStatus === "ok" ? "API online" : "API offline"}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="main">
        {tab === "overview" && <Overview metadata={metadata} apiError={apiStatus === "err" ? apiError : null} />}
        {tab === "verify" && <VerifyClaim />}
        {tab === "evidence" && <EvidenceExplorer />}
        {tab === "training" && <TrainingArtifacts />}
        {tab === "results" && <ResearchResults />}
      </main>
    </div>
  );
}
