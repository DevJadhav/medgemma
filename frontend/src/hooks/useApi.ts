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
} from '@/types/api';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

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
    const error: APIError = await response.json().catch(() => ({
      error: 'Unknown error',
      detail: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new Error(error.detail || error.error);
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
