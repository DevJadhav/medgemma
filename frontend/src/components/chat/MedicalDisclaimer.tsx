'use client';

import React, { useState } from 'react';
import { Alert } from '@/components/ui/Alert';

export interface MedicalDisclaimerProps {
  collapsible?: boolean;
  className?: string;
}

export function MedicalDisclaimer({
  collapsible = false,
  className = '',
}: MedicalDisclaimerProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const content = (
    <div className={!isExpanded ? 'hidden' : ''}>
      <p className="text-sm mb-2">
        <strong>Important:</strong> This AI assistant is{' '}
        <strong>not a substitute for professional medical advice</strong>,
        diagnosis, or treatment. Always consult with a qualified healthcare
        provider for medical concerns.
      </p>
      <p className="text-sm">
        <strong>Emergency:</strong> If you are experiencing a medical emergency,{' '}
        <strong>call 911</strong> or go to your nearest emergency room immediately.
      </p>
    </div>
  );

  if (collapsible) {
    return (
      <div className={className}>
        <Alert variant="warning" title="Medical Disclaimer">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-urgent-700 underline mb-2 block"
            aria-label="Toggle disclaimer"
          >
            {isExpanded ? 'Hide details' : 'Show details'}
          </button>
          {content}
        </Alert>
      </div>
    );
  }

  return (
    <div className={className}>
      <Alert variant="warning" title="Medical Disclaimer">
        {content}
      </Alert>
    </div>
  );
}
