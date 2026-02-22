'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import type {
  CommunicationRequest,
  CommunicationResponse,
  DiagnosticRequest,
  DiagnosticResponse,
  WorkflowRequest,
  WorkflowResponse,
  HealthResponse,
  APIError,
  EscalationItem,
  ReviewDecision,
  DashboardStats,
  SystemMetrics,
} from '@/types/api';

const API_BASE_URL = ''; // Always use relative paths to leverage Next.js rewrites or Nginx proxy

// Polling interval in milliseconds (30 seconds)
const POLLING_INTERVAL = 30000;

// =============================================================================
// Generic Fetch Wrapper
// =============================================================================

async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error: any = await response.json().catch(() => ({
      error: 'Unknown error',
    }));

    let errorMessage = error.error || `HTTP ${response.status}: ${response.statusText}`;

    if (error.detail) {
      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (Array.isArray(error.detail)) {
        errorMessage = error.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ');
      } else if (typeof error.detail === 'object') {
        errorMessage = JSON.stringify(error.detail);
      }
    }

    throw new Error(errorMessage);
  }

  return response.json();
}

// =============================================================================
// Health Check Hook
// =============================================================================

export function useHealthCheck() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<HealthResponse>('/health');
      setHealth(response);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { health, loading, error, checkHealth };
}

// =============================================================================
// Communication Hook
// =============================================================================

export function useCommunication() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (
    request: CommunicationRequest
  ): Promise<CommunicationResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<CommunicationResponse>(
        '/api/v1/communication/message',
        {
          method: 'POST',
          body: JSON.stringify(request),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { sendMessage, loading, error };
}

// =============================================================================
// Diagnostic Hook
// =============================================================================

export function useDiagnostic() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeImage = useCallback(async (
    request: DiagnosticRequest
  ): Promise<DiagnosticResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<DiagnosticResponse>(
        '/api/v1/diagnostic/analyze',
        {
          method: 'POST',
          body: JSON.stringify(request),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { analyzeImage, loading, error };
}

// =============================================================================
// Workflow Hook
// =============================================================================

export function useWorkflow() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processWorkflow = useCallback(async (
    request: WorkflowRequest
  ): Promise<WorkflowResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<WorkflowResponse>(
        '/api/v1/workflow/process',
        {
          method: 'POST',
          body: JSON.stringify(request),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Workflow failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { processWorkflow, loading, error };
}

// =============================================================================
// Orchestrator Hook
// =============================================================================

export function useOrchestrator() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processRequest = useCallback(async (
    message: string,
    patientId?: string
  ): Promise<Record<string, unknown> | null> => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ message });
      if (patientId) params.append('patient_id', patientId);

      const response = await fetchAPI<Record<string, unknown>>(
        `/api/v1/orchestrator/process?${params.toString()}`,
        { method: 'POST' }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { processRequest, loading, error };
}

// =============================================================================
// Escalations Hook with 30s Polling
// =============================================================================

interface EscalationsListResponse {
  escalations: EscalationItem[];
  total: number;
  timestamp: string;
}

interface EscalationFilters {
  priority?: 'high' | 'medium' | 'low';
  reason?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export function useEscalations(
  filters: EscalationFilters = {},
  enablePolling: boolean = true
) {
  const [escalations, setEscalations] = useState<EscalationItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const fetchEscalations = useCallback(async () => {
    if (!mountedRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (filters.priority) params.append('priority', filters.priority);
      if (filters.reason) params.append('reason', filters.reason);
      if (filters.status) params.append('status_filter', filters.status);
      if (filters.limit) params.append('limit', filters.limit.toString());
      if (filters.offset) params.append('offset', filters.offset.toString());

      const queryString = params.toString();
      const endpoint = queryString ? `/api/v1/escalations?${queryString}` : '/api/v1/escalations';

      const response = await fetchAPI<EscalationsListResponse>(endpoint);

      if (mountedRef.current) {
        // Convert timestamp strings to Date objects
        const escalationsWithDates = response.escalations.map(esc => ({
          ...esc,
          timestamp: new Date(esc.timestamp as unknown as string),
          reviewed_at: esc.reviewed_at ? new Date(esc.reviewed_at as unknown as string) : undefined,
        }));

        setEscalations(escalationsWithDates);
        setTotal(response.total);
        setLastUpdated(new Date());
      }

      return response;
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch escalations');
      }
      return null;
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [filters.priority, filters.reason, filters.status, filters.limit, filters.offset]);

  // Start/stop polling
  useEffect(() => {
    mountedRef.current = true;

    // Initial fetch
    fetchEscalations();

    // Set up polling
    if (enablePolling) {
      pollingRef.current = setInterval(fetchEscalations, POLLING_INTERVAL);
    }

    // Cleanup
    return () => {
      mountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [fetchEscalations, enablePolling]);

  // Manual refresh
  const refresh = useCallback(() => {
    return fetchEscalations();
  }, [fetchEscalations]);

  return {
    escalations,
    total,
    loading,
    error,
    lastUpdated,
    refresh,
  };
}

// =============================================================================
// Single Escalation Hook
// =============================================================================

export function useEscalation(escalationId: string | null) {
  const [escalation, setEscalation] = useState<EscalationItem | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchEscalation = useCallback(async () => {
    if (!escalationId) {
      setEscalation(null);
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetchAPI<EscalationItem>(
        `/api/v1/escalations/${escalationId}`
      );

      // Convert timestamps
      const escalationWithDates = {
        ...response,
        timestamp: new Date(response.timestamp as unknown as string),
        reviewed_at: response.reviewed_at ? new Date(response.reviewed_at as unknown as string) : undefined,
      };

      setEscalation(escalationWithDates);
      return escalationWithDates;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch escalation');
      return null;
    } finally {
      setLoading(false);
    }
  }, [escalationId]);

  useEffect(() => {
    fetchEscalation();
  }, [fetchEscalation]);

  return { escalation, loading, error, refresh: fetchEscalation };
}

// =============================================================================
// Review Submission Hook
// =============================================================================

export function useReviewSubmission() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitReview = useCallback(async (
    escalationId: string,
    decision: ReviewDecision
  ): Promise<EscalationItem | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetchAPI<EscalationItem>(
        `/api/v1/escalations/${escalationId}/review`,
        {
          method: 'POST',
          body: JSON.stringify({
            decision: decision.decision,
            notes: decision.notes,
            reviewer_id: 'current-user', // TODO: Get from auth context
            modified_response: decision.modified_response,
          }),
        }
      );

      return {
        ...response,
        timestamp: new Date(response.timestamp as unknown as string),
        reviewed_at: response.reviewed_at ? new Date(response.reviewed_at as unknown as string) : undefined,
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit review');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { submitReview, loading, error };
}

// =============================================================================
// Escalation Stats Hook with 30s Polling
// =============================================================================

export function useEscalationStats(enablePolling: boolean = true) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const fetchStats = useCallback(async () => {
    if (!mountedRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetchAPI<DashboardStats>('/api/v1/escalations/stats');

      if (mountedRef.current) {
        setStats(response);
      }

      return response;
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch stats');
      }
      return null;
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    // Initial fetch
    fetchStats();

    // Set up polling
    if (enablePolling) {
      pollingRef.current = setInterval(fetchStats, POLLING_INTERVAL);
    }

    return () => {
      mountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [fetchStats, enablePolling]);

  return { stats, loading, error, refresh: fetchStats };
}

// =============================================================================
// System Metrics Hook with 10s Polling for Real-Time Dashboard
// =============================================================================

export function useSystemMetrics(enablePolling: boolean = true) {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const fetchMetrics = useCallback(async () => {
    if (!mountedRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetchAPI<SystemMetrics>('/api/v1/system/metrics');

      if (mountedRef.current) {
        setMetrics(response);
      }

      return response;
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch system metrics');
      }
      return null;
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    // Initial fetch
    fetchMetrics();

    // Set up faster polling (10s) for real-time metrics
    if (enablePolling) {
      pollingRef.current = setInterval(fetchMetrics, 10000);
    }

    return () => {
      mountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [fetchMetrics, enablePolling]);

  return { metrics, loading, error, refresh: fetchMetrics };
}
// =============================================================================
// Compliance & Guardrails Hooks
// =============================================================================

export interface ComplianceStatus {
  status: string;
  timestamp: string;
  safeguards: Record<string, {
    status: string;
    details: string;
  }>;
}

export interface GuardrailsConfig {
  scope_patterns: Record<string, string[]>;
  jailbreak_categories: string[];
}

export interface GuardrailsTestResponse {
  sanitized_input: string;
  is_safe: boolean;
  is_valid_request: boolean;
  jailbreak: {
    detected: boolean;
    category: string | null;
    severity: string;
    risk_score: number;
    recommendation: string;
  };
  injection: {
    detected: boolean;
    risk_score: number;
    reason: string;
  };
  scope: {
    is_valid: boolean;
    scope: string | null;
    reason: string | null;
  };
}

export function useComplianceStatus() {
  const [status, setStatus] = useState<ComplianceStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<ComplianceStatus>('/api/v1/guardrails/compliance/status');
      setStatus(response);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch compliance status');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return { status, loading, error, refresh: fetchStatus };
}

export function useGuardrailsConfig() {
  const [config, setConfig] = useState<GuardrailsConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchAPI<GuardrailsConfig>('/api/v1/guardrails/config');
        setConfig(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch guardrails config');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return { config, loading, error };
}

export function useGuardrailsTest() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testGuardrails = useCallback(async (text: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<GuardrailsTestResponse>(
        '/api/v1/guardrails/test',
        {
          method: 'POST',
          body: JSON.stringify({ text }),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { testGuardrails, loading, error };
}

// =============================================================================
// Settings Hook - Backend Sync
// =============================================================================

export interface BackendSettings {
  model: string;
  inference_backend: string;
  training_strategy: string;
  available_models: string[];
  available_backends: string[];
  available_training_strategies: string[];
  gpu_available: boolean;
  environment: string;
}

export interface SettingsUpdateRequest {
  model?: string;
  inference_backend?: string;
  training_strategy?: string;
}

export function useSettings() {
  const [settings, setSettings] = useState<BackendSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);

  const fetchSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<BackendSettings>('/api/v1/settings');
      setSettings(response);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch settings');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateSettings = useCallback(async (updates: SettingsUpdateRequest) => {
    setSaving(true);
    setError(null);
    try {
      const response = await fetchAPI<BackendSettings>(
        '/api/v1/settings',
        {
          method: 'PUT',
          body: JSON.stringify(updates),
        }
      );
      setSettings(response);
      setLastSaved(new Date());
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update settings');
      return null;
    } finally {
      setSaving(false);
    }
  }, []);

  // Fetch on mount
  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  return {
    settings,
    loading,
    saving,
    error,
    lastSaved,
    fetchSettings,
    updateSettings,
  };
}

// =============================================================================
// RAG Hook - Knowledge Base Integration
// =============================================================================

export interface RAGDocument {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  relevance_score?: number;
}

export interface RAGQueryResponse {
  answer: string;
  confidence: number;
  sources: RAGDocument[];
  citations: Array<{
    text: string;
    source_id: string;
    relevance: number;
  }>;
  processing_time_ms: number;
  disclaimer?: string;
}

export interface RAGRetrieveResponse {
  documents: RAGDocument[];
  total: number;
  query: string;
}

export interface RAGStats {
  total_documents: number;
  total_chunks: number;
  embedding_model: string;
  vector_store_type: string;
  last_updated?: string;
}

export function useRAG() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (
    queryText: string,
    options?: {
      top_k?: number;
      include_sources?: boolean;
      filter_metadata?: Record<string, unknown>;
    }
  ): Promise<RAGQueryResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<RAGQueryResponse>(
        '/api/v1/rag/query',
        {
          method: 'POST',
          body: JSON.stringify({
            query: queryText,
            top_k: options?.top_k ?? 5,
            include_sources: options?.include_sources ?? true,
            filter_metadata: options?.filter_metadata,
          }),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'RAG query failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const retrieve = useCallback(async (
    queryText: string,
    topK: number = 5
  ): Promise<RAGRetrieveResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<RAGRetrieveResponse>(
        '/api/v1/rag/retrieve',
        {
          method: 'POST',
          body: JSON.stringify({
            query: queryText,
            top_k: topK,
          }),
        }
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Document retrieval failed');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { query, retrieve, loading, error };
}

export function useRAGStats() {
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<RAGStats>('/api/v1/rag/stats');
      setStats(response);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch RAG stats');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return { stats, loading, error, refresh: fetchStats };
}

// =============================================================================
// Inference Status Hook
// =============================================================================

export interface InferenceStatus {
  status: string;
  model_loaded: boolean;
  model_path?: string;
  model_source?: string;
  gpu_name?: string;
  gpu_memory_gb?: number;
  is_trained_model?: boolean;
}

export function useInferenceStatus() {
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchAPI<InferenceStatus>('/api/v1/inference/status');
      setStatus(response);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch inference status');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return { status, loading, error, refresh: fetchStatus };
}
