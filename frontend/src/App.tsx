import { useEffect, useState } from "react";
import { getHealth, getMetadata } from "./api";
import VerifyClaim from "./components/VerifyClaim";
import type { HealthResponse, MetadataResponse } from "./types";

type ServiceState = {
  ready: boolean;
  loading: boolean;
  health: HealthResponse | null;
  metadata: MetadataResponse | null;
  error: string | null;
};

const INITIAL_STATE: ServiceState = {
  ready: false,
  loading: true,
  health: null,
  metadata: null,
  error: null,
};

export default function App() {
  const [service, setService] = useState<ServiceState>(INITIAL_STATE);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [healthResult, metadataResult] = await Promise.allSettled([
        getHealth(),
        getMetadata(),
      ]);

      if (cancelled) return;

      const health = healthResult.status === "fulfilled" ? healthResult.value : null;
      const metadata = metadataResult.status === "fulfilled" ? metadataResult.value : null;

      setService({
        ready: Boolean(health),
        loading: false,
        health,
        metadata,
        error:
          healthResult.status === "rejected"
            ? healthResult.reason instanceof Error
              ? healthResult.reason.message
              : String(healthResult.reason)
            : metadataResult.status === "rejected"
              ? metadataResult.reason instanceof Error
                ? metadataResult.reason.message
                : String(metadataResult.reason)
              : null,
      });
    }

    load().catch((error: unknown) => {
      if (cancelled) return;
      setService({
        ready: false,
        loading: false,
        health: null,
        metadata: null,
        error: error instanceof Error ? error.message : String(error),
      });
    });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8.5l3.5 3.5L13 4" />
            </svg>
          </div>
          <div>
            <div className="brand-name">Veritas</div>
            <div className="brand-subtitle">Evidence-backed claim verification</div>
          </div>
        </div>
        <div className={`status-chip ${service.ready ? "status-ready" : service.loading ? "status-loading" : "status-offline"}`}>
          <span className="status-dot" />
          {service.loading ? "Checking backend" : service.ready ? "Backend ready" : "Backend offline"}
        </div>
      </header>

      <main className="page">
        <VerifyClaim service={service} />
      </main>
    </div>
  );
}
