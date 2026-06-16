export interface EvidenceItem {
  doc_id: string;
  text: string;
  title?: string;
  score?: number;
  citation_id?: number;
  relation?: "SUPPORTS" | "REFUTES" | "NEUTRAL";
  source?: string;
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

export interface HealthResponse {
  status: "ok";
  max_claim_length: number;
  verifier_backend: string;
  fallback_used: boolean;
  retrieval_backend: string;
  embedding_model: string | null;
  retrieval_fallback_used: boolean;
  reranker_backend: string;
  cross_encoder_model: string | null;
  reranker_fallback_used: boolean;
  checkpoint_path: string | null;
  model_name: string;
  verifier_macro_f1: number | null;
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
  artifact_checks: Record<string, unknown>;
}
