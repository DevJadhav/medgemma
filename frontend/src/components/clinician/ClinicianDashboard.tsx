'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { EscalationCard } from './EscalationCard';
import { ReviewModal } from './ReviewModal';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { useEscalations, useEscalationStats, useReviewSubmission } from '@/hooks/useApi';
import type { EscalationItem, ReviewDecision } from '@/types/api';

type PriorityFilter = 'all' | 'high' | 'medium' | 'low';
type CaseTypeFilter = 'all' | 'diagnostic' | 'workflow' | 'communication';

export function ClinicianDashboard() {
  // Use hooks with 30s polling
  const {
    escalations,
    total,
    loading,
    error: fetchError,
    lastUpdated,
    refresh
  } = useEscalations({}, true);

  const { stats, loading: statsLoading } = useEscalationStats(true);
  const { submitReview, loading: reviewLoading, error: reviewError } = useReviewSubmission();

  const [selectedEscalation, setSelectedEscalation] = useState<EscalationItem | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [priorityFilter, setPriorityFilter] = useState<PriorityFilter>('all');
  const [caseTypeFilter, setCaseTypeFilter] = useState<CaseTypeFilter>('all');
  const [localError, setLocalError] = useState<string | null>(null);

  // Combine errors
  const error = localError || fetchError || reviewError;

  const handleReview = (escalation: EscalationItem) => {
    setSelectedEscalation(escalation);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setSelectedEscalation(null);
  };

  const handleApprove = async (id: string, notes: string) => {
    const decision: ReviewDecision = {
      escalation_id: id,
      decision: 'approve',
      notes,
    };

    const result = await submitReview(id, decision);
    if (result) {
      handleCloseModal();
      refresh(); // Refresh list after approval
    }
  };

  const handleReject = async (id: string, reason: string) => {
    const decision: ReviewDecision = {
      escalation_id: id,
      decision: 'reject',
      notes: reason,
    };

    const result = await submitReview(id, decision);
    if (result) {
      handleCloseModal();
      refresh(); // Refresh list after rejection
    }
  };

  // Filter escalations locally
  const filteredEscalations = escalations.filter(e => {
    if (priorityFilter !== 'all' && e.priority !== priorityFilter) return false;
    if (caseTypeFilter !== 'all') {
      // Map agent_type to case type if available
      const agentType = (e as any).agent_type;
      if (agentType && agentType !== caseTypeFilter) return false;
    }
    return true;
  });

  // Count by priority from local data or stats
  const highCount = stats?.pending_reviews || escalations.filter(e => e.priority === 'high').length;
  const pendingCount = stats?.pending_reviews || total;
  const criticalToday = stats?.critical_findings_today || 0;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Clinician Review Dashboard
              </h1>
              {lastUpdated && (
                <p className="text-sm text-gray-500">
                  Last updated: {lastUpdated.toLocaleTimeString()} • Auto-refreshing every 30s
                </p>
              )}
            </div>
            <Button
              variant="outline"
              onClick={refresh}
              disabled={loading}
              aria-label="Refresh"
            >
              {loading ? 'Refreshing...' : 'Refresh Now'}
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Pending Reviews</p>
                  <p
                    data-testid="pending-count"
                    className="text-2xl font-bold text-gray-900"
                  >
                    {pendingCount}
                  </p>
                </div>
                <Badge variant="info" size="lg">Queue</Badge>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Critical Today</p>
                  <p className="text-2xl font-bold text-emergency-600">
                    {criticalToday}
                  </p>
                </div>
                <Badge variant="emergency" size="lg" pulse={criticalToday > 0}>
                  Critical
                </Badge>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">High Priority</p>
                  <p className="text-2xl font-bold text-urgent-600">{highCount}</p>
                </div>
                <Badge variant="urgent" size="lg">High</Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 mb-6">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Priority:</span>
            {(['all', 'high', 'medium', 'low'] as PriorityFilter[]).map(
              priority => (
                <Button
                  key={priority}
                  size="sm"
                  variant={priorityFilter === priority ? 'default' : 'outline'}
                  onClick={() => setPriorityFilter(priority)}
                >
                  {priority.charAt(0).toUpperCase() + priority.slice(1)}
                </Button>
              )
            )}
          </div>
          <div className="flex items-center gap-2 ml-4">
            <span className="text-sm text-gray-500">Type:</span>
            {(
              ['all', 'diagnostic', 'workflow', 'communication'] as CaseTypeFilter[]
            ).map(type => (
              <Button
                key={type}
                size="sm"
                variant={caseTypeFilter === type ? 'default' : 'outline'}
                onClick={() => setCaseTypeFilter(type)}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </Button>
            ))}
          </div>
        </div>

        {/* Error State */}
        {error && (
          <Alert variant="error" className="mb-6" dismissible onDismiss={() => setLocalError(null)}>
            {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
          </div>
        )}

        {/* Empty State */}
        {!loading && filteredEscalations.length === 0 && (
          <Card>
            <CardContent className="p-8 text-center">
              <p className="text-gray-500">No pending escalations</p>
              <p className="text-sm text-gray-400 mt-1">
                All cases have been reviewed or no cases match your filters
              </p>
            </CardContent>
          </Card>
        )}

        {/* Escalation List */}
        {!loading && filteredEscalations.length > 0 && (
          <div className="grid gap-4">
            {filteredEscalations.map(escalation => (
              <EscalationCard
                key={escalation.id}
                escalation={escalation}
                onReview={handleReview}
              />
            ))}
          </div>
        )}
      </main>

      {/* Review Modal */}
      <ReviewModal
        escalation={selectedEscalation}
        open={modalOpen}
        onClose={handleCloseModal}
        onApprove={handleApprove}
        onReject={handleReject}
        loading={reviewLoading}
      />
    </div>
  );
}
