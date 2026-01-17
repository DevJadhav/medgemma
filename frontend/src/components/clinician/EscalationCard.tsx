'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
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

const caseTypeLabels: Record<string, string> = {
  diagnostic: 'Diagnostic',
  workflow: 'Workflow',
  communication: 'Communication',
};

export interface EscalationCardProps {
  escalation: EscalationItem;
  onReview: (escalation: EscalationItem) => void;
}

export function EscalationCard({ escalation, onReview }: EscalationCardProps) {
  const formattedTime = new Date(escalation.timestamp).toLocaleString();
  const confidencePercent = Math.round(escalation.confidence * 100);

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <Badge
              data-testid="priority-badge"
              variant={priorityVariants[escalation.priority]}
              pulse={escalation.priority === 'critical'}
            >
              {priorityLabels[escalation.priority]}
            </Badge>
            <Badge variant="default">
              {caseTypeLabels[escalation.case_type]}
            </Badge>
          </div>
          <span
            data-testid="escalation-timestamp"
            className="text-xs text-gray-500"
          >
            {formattedTime}
          </span>
        </div>

        <div className="mb-3">
          <p className="text-sm font-medium text-gray-900 mb-1">
            Patient: {escalation.patient_id}
          </p>
          <p className="text-sm text-gray-700">{escalation.reason}</p>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">AI Confidence:</span>
            <span
              className={`text-sm font-medium ${
                confidencePercent < 70
                  ? 'text-emergency-600'
                  : confidencePercent < 85
                  ? 'text-urgent-600'
                  : 'text-success-600'
              }`}
            >
              {confidencePercent}%
            </span>
          </div>
          <Button size="sm" onClick={() => onReview(escalation)}>
            Review
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
