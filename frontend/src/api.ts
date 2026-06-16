import type {
  EvidenceItem,
  ExplainResponse,
  MetadataResponse,
  PipelineResponse,
  RetrieveResponse,
  VerifyResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail) ?? "Request failed"
    );
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail) ?? "Request failed"
    );
  }
  return res.json();
}

export function runPipeline(claim: string, topK = 5): Promise<PipelineResponse> {
  return post("/pipeline", { claim, top_k: topK });
}

export function runVerify(claim: string, topK = 5): Promise<VerifyResponse> {
  return post("/verify", { claim, top_k: topK });
}

export function runRetrieve(claim: string, topK = 5): Promise<RetrieveResponse> {
  return post("/retrieve", { claim, top_k: topK });
}

export function runExplain(
  claim: string,
  label: string,
  evidence: EvidenceItem[]
): Promise<ExplainResponse> {
  return post("/explain", { claim, label, evidence });
}

export function fetchMetadata(): Promise<MetadataResponse> {
  return get("/metadata");
}

export function fetchHealth(): Promise<{ status: string }> {
  return get("/health");
}
