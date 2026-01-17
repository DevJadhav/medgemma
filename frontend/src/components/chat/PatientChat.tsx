'use client';

import React, { useRef, useEffect } from 'react';
import { useChat } from '@/hooks/useChat';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { MedicalDisclaimer } from './MedicalDisclaimer';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Alert, EmergencyAlert } from '@/components/ui/Alert';
import type { TriageLevel } from '@/types/api';

const triageLabels: Record<TriageLevel, string> = {
  EMERGENCY: 'Emergency',
  URGENT: 'Urgent',
  SOON: 'Soon',
  ROUTINE: 'Routine',
  INFORMATIONAL: 'Informational',
};

const triageBadgeVariants: Record<TriageLevel, 'emergency' | 'urgent' | 'success' | 'info' | 'default'> = {
  EMERGENCY: 'emergency',
  URGENT: 'urgent',
  SOON: 'info',
  ROUTINE: 'success',
  INFORMATIONAL: 'default',
};

export interface PatientChatProps {
  patientId?: string;
  className?: string;
}

export function PatientChat({ patientId, className = '' }: PatientChatProps) {
  const {
    messages,
    currentTriageLevel,
    isEmergency,
    loading,
    error,
    sendMessage,
  } = useChat(patientId);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const welcomeMessage = {
    id: 'welcome',
    role: 'assistant' as const,
    content: "Hello! How can I help you today? Please describe your symptoms or health concerns, and I'll do my best to assist you.",
    timestamp: new Date().toISOString(),
  };

  const displayMessages = messages.length === 0 ? [welcomeMessage] : messages;

  return (
    <div data-testid="patient-chat" className={`flex flex-col h-full ${className}`}>
      {/* Header with triage indicator */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h1 className="text-xl font-semibold text-gray-900">
          Health Assistant
        </h1>
        {currentTriageLevel && (
          <div data-testid="triage-indicator">
            <Badge
              variant={triageBadgeVariants[currentTriageLevel]}
              pulse={currentTriageLevel === 'EMERGENCY'}
            >
              {triageLabels[currentTriageLevel]}
            </Badge>
          </div>
        )}
      </div>

      {/* Emergency alert banner */}
      {isEmergency && (
        <div className="px-4 pt-4">
          <EmergencyAlert>
            This appears to be a medical emergency. Please call 911 or go to your
            nearest emergency room immediately.
          </EmergencyAlert>
        </div>
      )}

      {/* Medical disclaimer */}
      <div className="px-4 pt-4">
        <MedicalDisclaimer collapsible />
      </div>

      {/* Error message */}
      {error && (
        <div className="px-4 pt-4">
          <Alert variant="error" title="Error">
            {error}
          </Alert>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-4">
          {displayMessages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <ChatInput
          onSend={sendMessage}
          loading={loading}
          disabled={false}
          placeholder="Describe your symptoms or ask a health question..."
        />
        <p className="text-xs text-gray-500 mt-2 text-center">
          Press Enter to send • Your conversation is private and secure
        </p>
      </div>
    </div>
  );
}
