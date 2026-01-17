import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ClinicianDashboard } from '@/components/clinician/ClinicianDashboard';
import { EscalationCard } from '@/components/clinician/EscalationCard';
import { ReviewModal } from '@/components/clinician/ReviewModal';
import type { EscalationItem } from '@/types/api';

// =============================================================================
// Mock Data
// =============================================================================

const mockEscalation: EscalationItem = {
  id: 'esc-001',
  patient_id: 'patient-123',
  case_type: 'diagnostic',
  reason: 'Critical finding: Possible pneumothorax detected',
  confidence: 0.72,
  timestamp: '2024-01-15T10:00:00Z',
  priority: 'high',
  status: 'pending',
  context: {
    image_type: 'chest_xray',
    findings: ['Possible pneumothorax', 'No cardiomegaly'],
  },
};

const mockEscalations: EscalationItem[] = [
  mockEscalation,
  {
    id: 'esc-002',
    patient_id: 'patient-456',
    case_type: 'communication',
    reason: 'Safety concern: Patient expressed suicidal ideation',
    confidence: 0.65,
    timestamp: '2024-01-15T09:30:00Z',
    priority: 'critical',
    status: 'pending',
    context: {
      conversation_excerpt: 'I feel like giving up...',
    },
  },
  {
    id: 'esc-003',
    patient_id: 'patient-789',
    case_type: 'workflow',
    reason: 'Prior authorization requires manual review',
    confidence: 0.85,
    timestamp: '2024-01-15T08:00:00Z',
    priority: 'medium',
    status: 'pending',
    context: {
      procedure: 'MRI Brain with contrast',
      insurance: 'Medicare',
    },
  },
];

// =============================================================================
// EscalationCard Tests
// =============================================================================

describe('EscalationCard', () => {
  it('renders escalation details', () => {
    render(
      <EscalationCard
        escalation={mockEscalation}
        onReview={jest.fn()}
      />
    );
    
    expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    expect(screen.getByText(/patient-123/i)).toBeInTheDocument();
    expect(screen.getByText('72%')).toBeInTheDocument();
  });

  it('shows priority badge', () => {
    render(
      <EscalationCard
        escalation={mockEscalation}
        onReview={jest.fn()}
      />
    );
    
    expect(screen.getByText('High')).toBeInTheDocument();
  });

  it('shows critical badge with pulse animation', () => {
    const criticalEscalation = { ...mockEscalation, priority: 'critical' as const };
    render(
      <EscalationCard
        escalation={criticalEscalation}
        onReview={jest.fn()}
      />
    );
    
    expect(screen.getByTestId('priority-badge')).toHaveClass('animate-pulse');
  });

  it('shows case type indicator', () => {
    render(
      <EscalationCard
        escalation={mockEscalation}
        onReview={jest.fn()}
      />
    );
    
    expect(screen.getByText('Diagnostic')).toBeInTheDocument();
  });

  it('calls onReview when review button is clicked', async () => {
    const handleReview = jest.fn();
    render(
      <EscalationCard
        escalation={mockEscalation}
        onReview={handleReview}
      />
    );
    
    await userEvent.click(screen.getByRole('button', { name: /review/i }));
    expect(handleReview).toHaveBeenCalledWith(mockEscalation);
  });

  it('displays formatted timestamp', () => {
    render(
      <EscalationCard
        escalation={mockEscalation}
        onReview={jest.fn()}
      />
    );
    
    expect(screen.getByTestId('escalation-timestamp')).toBeInTheDocument();
  });
});

// =============================================================================
// ReviewModal Tests
// =============================================================================

describe('ReviewModal', () => {
  it('renders modal with escalation details', () => {
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={jest.fn()}
        onApprove={jest.fn()}
        onReject={jest.fn()}
      />
    );
    
    expect(screen.getByText(/review case/i)).toBeInTheDocument();
    expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={false}
        onClose={jest.fn()}
        onApprove={jest.fn()}
        onReject={jest.fn()}
      />
    );
    
    expect(screen.queryByText(/review case/i)).not.toBeInTheDocument();
  });

  it('calls onApprove with notes when approved', async () => {
    const handleApprove = jest.fn();
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={jest.fn()}
        onApprove={handleApprove}
        onReject={jest.fn()}
      />
    );
    
    const notesInput = screen.getByPlaceholderText(/add clinical notes/i);
    await userEvent.type(notesInput, 'Confirmed findings');
    await userEvent.click(screen.getByRole('button', { name: /approve/i }));
    
    expect(handleApprove).toHaveBeenCalledWith(mockEscalation.id, 'Confirmed findings');
  });

  it('calls onReject with reason when rejected', async () => {
    const handleReject = jest.fn();
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={jest.fn()}
        onApprove={jest.fn()}
        onReject={handleReject}
      />
    );
    
    const notesInput = screen.getByPlaceholderText(/add clinical notes/i);
    await userEvent.type(notesInput, 'False positive');
    await userEvent.click(screen.getByRole('button', { name: /reject/i }));
    
    expect(handleReject).toHaveBeenCalledWith(mockEscalation.id, 'False positive');
  });

  it('calls onClose when cancel button is clicked', async () => {
    const handleClose = jest.fn();
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={handleClose}
        onApprove={jest.fn()}
        onReject={jest.fn()}
      />
    );
    
    await userEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(handleClose).toHaveBeenCalled();
  });

  it('shows context information', () => {
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={jest.fn()}
        onApprove={jest.fn()}
        onReject={jest.fn()}
      />
    );
    
    expect(screen.getByText(/chest_xray/i)).toBeInTheDocument();
  });

  it('disables buttons when loading', () => {
    render(
      <ReviewModal
        escalation={mockEscalation}
        open={true}
        onClose={jest.fn()}
        onApprove={jest.fn()}
        onReject={jest.fn()}
        loading={true}
      />
    );
    
    expect(screen.getByRole('button', { name: /approve/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /reject/i })).toBeDisabled();
  });
});

// =============================================================================
// ClinicianDashboard Tests
// =============================================================================

describe('ClinicianDashboard', () => {
  // Mock fetch for escalations
  beforeEach(() => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ escalations: mockEscalations }),
    });
  });

  it('renders dashboard title', async () => {
    render(<ClinicianDashboard />);
    expect(screen.getByText(/clinician review dashboard/i)).toBeInTheDocument();
  });

  it('displays escalation queue', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    });
  });

  it('shows stats summary', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('pending-count')).toHaveTextContent('3');
    });
  });

  it('filters by priority', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    });
    
    await userEvent.click(screen.getByRole('button', { name: /critical/i }));
    
    // Should only show critical priority items
    await waitFor(() => {
      expect(screen.getByText(/suicidal ideation/i)).toBeInTheDocument();
      expect(screen.queryByText(/pneumothorax/i)).not.toBeInTheDocument();
    });
  });

  it('filters by case type', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    });
    
    await userEvent.click(screen.getByRole('button', { name: /diagnostic/i }));
    
    // Should only show diagnostic type items
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
      expect(screen.queryByText(/suicidal ideation/i)).not.toBeInTheDocument();
    });
  });

  it('opens review modal when reviewing an escalation', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    });
    
    const reviewButtons = screen.getAllByRole('button', { name: /review/i });
    await userEvent.click(reviewButtons[0]);
    
    expect(screen.getByText(/review case/i)).toBeInTheDocument();
  });

  it('shows empty state when no escalations', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ escalations: [] }),
    });
    
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/no pending escalations/i)).toBeInTheDocument();
    });
  });

  it('shows error state on fetch failure', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));
    
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/failed to load escalations/i)).toBeInTheDocument();
    });
  });

  it('refreshes escalations when refresh button is clicked', async () => {
    render(<ClinicianDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/pneumothorax/i)).toBeInTheDocument();
    });
    
    await userEvent.click(screen.getByRole('button', { name: /refresh/i }));
    
    // Should have called fetch twice
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});
