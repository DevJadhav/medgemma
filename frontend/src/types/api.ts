/**
 * API Types for MedAI Compass Frontend
 * These types mirror the FastAPI backend models
 */

// =============================================================================
// Triage and Urgency Types
// =============================================================================

export type TriageLevel =
  | 'EMERGENCY'
  | 'URGENT'
  | 'SOON'
  | 'ROUTINE'
  | 'INFORMATIONAL';

export const TRIAGE_COLORS: Record<TriageLevel, { bg: string; text: string; border: string }> = {
  EMERGENCY: { bg: 'bg-emergency-100', text: 'text-emergency-700', border: 'border-emergency-500' },
  URGENT: { bg: 'bg-urgent-100', text: 'text-urgent-600', border: 'border-urgent-500' },
  SOON: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-500' },
  ROUTINE: { bg: 'bg-primary-100', text: 'text-primary-700', border: 'border-primary-500' },
  INFORMATIONAL: { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-500' },
};

export const TRIAGE_LABELS: Record<TriageLevel, string> = {
  EMERGENCY: 'Emergency - Call 911',
  URGENT: 'Urgent - Same Day',
  SOON: 'Soon - 24-48 Hours',
  ROUTINE: 'Routine',
  INFORMATIONAL: 'Informational',
};

// =============================================================================
// Communication Types
// =============================================================================

export interface CommunicationRequest {
  message: string;
  patient_id?: string;
  session_id?: string;
  language?: string;
}

export interface CommunicationResponse {
  request_id: string;
  response: string;
  triage_level: TriageLevel;
  requires_escalation: boolean;
  disclaimer: string;
  session_id: string;
  processing_time_ms: number;
}

export interface ChatMessage {
  id: string;
  role: 'patient' | 'assistant' | 'user';
  content: string;
  timestamp: Date | string;
  triage_level?: TriageLevel;
  triageLevel?: TriageLevel; // Alias for camelCase usage
  requires_escalation?: boolean;
  disclaimer?: string;
}

// =============================================================================
// Diagnostic Types
// =============================================================================

export interface DiagnosticRequest {
  image_path?: string;
  image_base64?: string;
  image_type: 'cxr' | 'ct' | 'mri' | 'pathology';
  patient_id?: string;
  clinical_context?: string;
}

export interface DiagnosticFinding {
  finding: string;
  location?: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  bounding_box?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface DiagnosticResponse {
  request_id: string;
  status: 'pending' | 'completed' | 'error';
  findings: DiagnosticFinding[];
  confidence: number;
  report: string;
  requires_review: boolean;
  processing_time_ms: number;
}

// =============================================================================
// Workflow Types
// =============================================================================

export type WorkflowType = 'scheduling' | 'documentation' | 'prior_auth';

export interface WorkflowRequest {
  request_type: WorkflowType;
  patient_id?: string;
  encounter_id?: string;
  data: Record<string, unknown>;
}

export interface WorkflowResponse {
  request_id: string;
  status: 'completed' | 'error';
  result: Record<string, unknown>;
  processing_time_ms: number;
}

// =============================================================================
// Escalation Types
// =============================================================================

export type EscalationReason =
  | 'critical_finding'
  | 'low_confidence'
  | 'safety_concern'
  | 'manual_request';

export interface EscalationItem {
  id: string;
  request_id: string;
  timestamp: Date | string;
  patient_id?: string;
  reason: EscalationReason;
  priority: 'high' | 'medium' | 'low' | 'critical';
  status: 'pending' | 'in_review' | 'approved' | 'rejected';

  // Diagnostic escalation
  diagnostic_result?: DiagnosticResponse;

  // Communication escalation
  communication_result?: CommunicationResponse;
  original_message?: string;

  // Additional fields from backend
  agent_type?: string;
  case_type?: string;
  confidence?: number;
  confidence_score?: number;
  uncertainty_score?: number;
  context?: string;

  // Reviewer info
  assigned_to?: string;
  reviewed_by?: string;
  reviewed_at?: Date | string;
  review_notes?: string;
  modified_response?: string;
}

export interface ReviewDecision {
  escalation_id: string;
  decision: 'approve' | 'reject' | 'modify';
  notes: string;
  modified_response?: string;
}

// =============================================================================
// Health Check Types
// =============================================================================

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  services: Record<string, 'healthy' | 'unhealthy' | 'unavailable'>;
}

// =============================================================================
// Session Types
// =============================================================================

export interface SessionData {
  session_id: string;
  patient_id?: string;
  created_at: Date;
  messages: ChatMessage[];
}

// =============================================================================
// Error Types
// =============================================================================

export interface APIError {
  error: string;
  detail?: string;
  request_id?: string;
}

// =============================================================================
// Statistics Types (for dashboards)
// =============================================================================

export interface DashboardStats {
  // Fields from EscalationStatsResponse
  total_pending: number;
  total_in_review: number;
  total_approved_today: number;
  total_rejected_today: number;
  average_review_time_ms: number;

  // Legacy fields (optional or to be computed client-side)
  total_requests?: number;
  pending_reviews?: number; // Mapped to total_pending in derived logic
  critical_findings_today?: number;
  average_response_time_ms?: number;
  escalation_rate?: number;
}
