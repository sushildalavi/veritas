import type { HealthResponse, MetadataResponse, PipelineResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function readErrorDetail(res: Response): Promise<string> {
  const text = await res.text();

  if (!text) return res.statusText || "Request failed";

  try {
    const parsed = JSON.parse(text) as { detail?: unknown };

    if (typeof parsed.detail === "string") return parsed.detail;
    if (parsed.detail) return JSON.stringify(parsed.detail);
  } catch {
    // Fall through to the raw response body.
  }

  return text;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!res.ok) {
    throw new Error(await readErrorDetail(res));
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
