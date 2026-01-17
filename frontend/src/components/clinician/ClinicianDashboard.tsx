'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { EscalationCard } from './EscalationCard';
import { ReviewModal } from './ReviewModal';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import type { EscalationItem } from '@/types/api';

type PriorityFilter = 'all' | 'critical' | 'high' | 'medium' | 'low';
type CaseTypeFilter = 'all' | 'diagnostic' | 'workflow' | 'communication';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

export function ClinicianDashboard() {
  const [escalations, setEscalations] = useState<EscalationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEscalation, setSelectedEscalation] = useState<EscalationItem | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [priorityFilter, setPriorityFilter] = useState<PriorityFilter>('all');
  const [caseTypeFilter, setCaseTypeFilter] = useState<CaseTypeFilter>('all');

  const fetchEscalations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/escalations`);
      if (!response.ok) {
        throw new Error('Failed to fetch escalations');
      }
      const data = await response.json();
      setEscalations(data.escalations || []);
    } catch (err) {
      setError('Failed to load escalations. Please try again.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEscalations();
  }, [fetchEscalations]);

  const handleReview = (escalation: EscalationItem) => {
    setSelectedEscalation(escalation);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setSelectedEscalation(null);
  };

  const handleApprove = async (id: string, notes: string) => {
    setReviewLoading(true);
    try {
      await fetch(`${API_BASE_URL}/api/v1/escalations/${id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision: 'approved', notes }),
      });
      setEscalations(prev => prev.filter(e => e.id !== id));
      handleCloseModal();
    } catch (err) {
      setError('Failed to approve. Please try again.');
    } finally {
      setReviewLoading(false);
    }
  };

  const handleReject = async (id: string, reason: string) => {
    setReviewLoading(true);
    try {
      await fetch(`${API_BASE_URL}/api/v1/escalations/${id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision: 'rejected', notes: reason }),
      });
      setEscalations(prev => prev.filter(e => e.id !== id));
      handleCloseModal();
    } catch (err) {
      setError('Failed to reject. Please try again.');
    } finally {
      setReviewLoading(false);
    }
  };

  // Filter escalations
  const filteredEscalations = escalations.filter(e => {
    if (priorityFilter !== 'all' && e.priority !== priorityFilter) return false;
    if (caseTypeFilter !== 'all' && e.case_type !== caseTypeFilter) return false;
    return true;
  });

  // Count by priority
  const criticalCount = escalations.filter(e => e.priority === 'critical').length;
  const highCount = escalations.filter(e => e.priority === 'high').length;
  const pendingCount = escalations.length;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">
              Clinician Review Dashboard
            </h1>
            <Button
              variant="outline"
              onClick={fetchEscalations}
              disabled={loading}
              aria-label="Refresh"
            >
              Refresh
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
                  <p className="text-sm text-gray-500">Critical Priority</p>
                  <p className="text-2xl font-bold text-emergency-600">
                    {criticalCount}
                  </p>
                </div>
                <Badge variant="emergency" size="lg" pulse={criticalCount > 0}>
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
            {(['all', 'critical', 'high', 'medium', 'low'] as PriorityFilter[]).map(
              priority => (
                <Button
                  key={priority}
                  size="sm"
                  variant={priorityFilter === priority ? 'primary' : 'outline'}
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
                variant={caseTypeFilter === type ? 'primary' : 'outline'}
                onClick={() => setCaseTypeFilter(type)}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </Button>
            ))}
          </div>
        </div>

        {/* Error State */}
        {error && (
          <Alert variant="error" className="mb-6" dismissible onDismiss={() => setError(null)}>
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
