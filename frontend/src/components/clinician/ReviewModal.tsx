'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import type { EscalationItem } from '@/types/api';

const priorityLabels: Record<string, string> = {
  critical: 'Critical',
  high: 'High',
  medium: 'Medium',
  low: 'Low',
};

const priorityVariants: Record<string, 'emergency' | 'urgent' | 'info' | 'default'> = {
  critical: 'emergency',
  high: 'urgent',
  medium: 'info',
  low: 'default',
};

export interface ReviewModalProps {
  escalation: EscalationItem | null;
  open: boolean;
  onClose: () => void;
  onApprove: (id: string, notes: string) => void;
  onReject: (id: string, reason: string) => void;
  loading?: boolean;
}

export function ReviewModal({
  escalation,
  open,
  onClose,
  onApprove,
  onReject,
  loading = false,
}: ReviewModalProps) {
  const [notes, setNotes] = useState('');

  if (!open || !escalation) {
    return null;
  }

  const confidencePercent = escalation.confidence != null
    ? Math.round(escalation.confidence * 100)
    : escalation.confidence_score != null
      ? Math.round(escalation.confidence_score * 100)
      : null;

  const handleApprove = () => {
    onApprove(escalation.id, notes);
    setNotes('');
  };

  const handleReject = () => {
    onReject(escalation.id, notes);
    setNotes('');
  };

  const handleCancel = () => {
    setNotes('');
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={handleCancel}
      />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="px-6 py-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900">
                Review Case
              </h2>
              <Badge
                variant={priorityVariants[escalation.priority] || 'default'}
                pulse={escalation.priority === 'critical' || escalation.priority === 'high'}
              >
                {priorityLabels[escalation.priority] || escalation.priority}
              </Badge>
            </div>
          </div>

          {/* Content */}
          <div className="px-6 py-4 space-y-4">
            {/* Patient Info */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">
                Patient ID
              </h3>
              <p className="text-gray-900">{escalation.patient_id || 'Unknown'}</p>
            </div>

            {/* Reason */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">
                Escalation Reason
              </h3>
              <p className="text-gray-900">{escalation.reason}</p>
            </div>

            {/* Confidence */}
            {confidencePercent !== null && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-1">
                  AI Confidence
                </h3>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${confidencePercent < 70
                          ? 'bg-emergency-500'
                          : confidencePercent < 85
                            ? 'bg-urgent-500'
                            : 'bg-success-500'
                        }`}
                      style={{ width: `${confidencePercent}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{confidencePercent}%</span>
                </div>
              </div>
            )}

            {/* Context */}
            {(escalation.context || escalation.original_message) && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-1">
                  Additional Context
                </h3>
                <pre className="text-sm bg-gray-50 p-3 rounded-lg overflow-x-auto">
                  {escalation.original_message || JSON.stringify(escalation.context, null, 2)}
                </pre>
              </div>
            )}

            {/* Notes Input */}
            <div>
              <label
                htmlFor="clinical-notes"
                className="block text-sm font-medium text-gray-500 mb-1"
              >
                Clinical Notes
              </label>
              <textarea
                id="clinical-notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add clinical notes or reasoning..."
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
            <Button
              variant="outline"
              onClick={handleCancel}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={loading}
              loading={loading}
            >
              Reject
            </Button>
            <Button
              onClick={handleApprove}
              disabled={loading}
              loading={loading}
            >
              Approve
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
