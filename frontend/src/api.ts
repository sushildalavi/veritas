import type { HealthResponse, MetadataResponse, PipelineResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail) || "Request failed"
    );
  }

  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson("/health");
}

export function getMetadata(): Promise<MetadataResponse> {
  return requestJson("/metadata");
}

export function checkClaim(claim: string, topK = 5): Promise<PipelineResponse> {
  return requestJson("/pipeline", {
    method: "POST",
    body: JSON.stringify({ claim, top_k: topK }),
  });
}
