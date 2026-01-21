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
  const [isExpanded, setIsExpanded] = useState(!collapsible);

  const content = (
    <div className={`${!isExpanded ? 'hidden' : ''} space-y-2`}>
      <p className="text-sm leading-relaxed">
        <strong>Important:</strong> This AI assistant is{' '}
        <strong>not a substitute for professional medical advice</strong>,
        diagnosis, or treatment. Always consult with a qualified healthcare
        provider for medical concerns.
      </p>
      <p className="text-sm leading-relaxed">
        <strong>🚨 Emergency:</strong> If you are experiencing a medical emergency,{' '}
        <strong>call 112</strong> or go to your nearest emergency room immediately.
      </p>
    </div>
  );

  if (collapsible) {
    return (
      <div className={className}>
        <div className="bg-amber-50 border-l-4 border-amber-500 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <h4 className="font-semibold text-amber-800">Medical Disclaimer</h4>
            </div>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs font-medium text-amber-700 hover:text-amber-900 px-2.5 py-1 rounded-lg hover:bg-amber-100 transition-colors"
              aria-label="Toggle disclaimer"
            >
              {isExpanded ? '▲ Hide' : '▼ Show details'}
            </button>
          </div>
          <div className="text-amber-800">{content}</div>
        </div>
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
