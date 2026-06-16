export interface EvidenceItem {
  doc_id: string;
  text: string;
  title?: string;
  score?: number;
  citation_id?: number;
}

export interface VerifyResponse {
  request_id: string;
  verdict: "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO";
  confidence: number;
  explanation: string;
  citation_valid: boolean;
  explanation_mode: string;
  backend_used: string;
  retrieval_backend: string;
  retrieval_fallback_used: boolean;
  reranker_backend: string;
  reranker_fallback_used: boolean;
  evidence: EvidenceItem[];
  fallback_used: boolean;
  latency_ms: number;
  model_name: string;
  verifier_macro_f1?: number;
}

export interface RetrieveResponse {
  claim: string;
  evidence: EvidenceItem[];
  retrieval_backend: string;
  latency_ms: number;
}

export interface ExplainResponse {
  explanation: string;
  citations: string[];
  backend_used: string;
  latency_ms: number;
}

export interface LatencyBreakdown {
  retrieval_ms: number;
  verification_ms: number;
  explanation_ms: number;
  total_ms: number;
}

export interface PipelineResponse {
  request_id: string;
  claim: string;
  verdict: "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO";
  confidence: number;
  evidence: EvidenceItem[];
  explanation: string;
  citations: string[];
  citation_valid: boolean;
  backend_used: string;
  retrieval_backend: string;
  explanation_mode: string;
  latency: LatencyBreakdown;
}

export interface MetadataResponse {
  project: string;
  version: string;
  description: string;
  verifier_checkpoint: string;
  verifier_model: string;
  verifier_oracle_macro_f1: number;
  verifier_retrieved_macro_f1: number;
  retrieval_recall_at_10: number;
  oracle_retrieved_gap: number;
  retrieval_profile: string;
  available_backends: string[];
  endpoints: string[];
  artifact_checks: Record<string, boolean | string>;
}

export type Tab =
  | "overview"
  | "verify"
  | "evidence"
  | "training"
  | "results";
