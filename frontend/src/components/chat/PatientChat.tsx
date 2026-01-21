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
    content: "Hello! 👋 I'm your health assistant. How can I help you today? Please describe your symptoms or health concerns, and I'll do my best to assist you.",
    timestamp: new Date().toISOString(),
  };

  const displayMessages = messages.length === 0 ? [welcomeMessage] : messages;

  return (
    <div data-testid="patient-chat" className={`flex flex-col h-full ${className}`}>
      {/* Header with triage indicator */}
      <div className="flex items-center justify-between p-5 border-b border-gray-100 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
            <span className="text-lg">💬</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">
              Health Assistant
            </h1>
            <p className="text-xs text-gray-500">Powered by MedGemma</p>
          </div>
        </div>
        {currentTriageLevel && (
          <div data-testid="triage-indicator">
            <Badge
              variant={triageBadgeVariants[currentTriageLevel]}
              pulse={currentTriageLevel === 'EMERGENCY'}
              size="md"
            >
              {triageLabels[currentTriageLevel]}
            </Badge>
          </div>
        )}
      </div>

      {/* Emergency alert banner */}
      {isEmergency && (
        <div className="px-5 pt-4">
          <EmergencyAlert>
            This appears to be a medical emergency. Please call 112 or go to your
            nearest emergency room immediately.
          </EmergencyAlert>
        </div>
      )}

      {/* Medical disclaimer */}
      <div className="px-5 pt-4">
        <MedicalDisclaimer collapsible />
      </div>

      {/* Error message */}
      {error && (
        <div className="px-5 pt-4">
          <Alert variant="error" title="Error">
            {error}
          </Alert>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-5 bg-gradient-to-b from-gray-50/50 to-white">
        <div className="space-y-4 max-w-3xl mx-auto">
          {displayMessages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
                  <span className="text-sm">🏥</span>
                </div>
                <div className="bg-white rounded-2xl px-5 py-4 border border-gray-100 shadow-sm">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="p-5 border-t border-gray-100 bg-white">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSend={sendMessage}
            loading={loading}
            disabled={false}
            placeholder="Describe your symptoms or ask a health question..."
          />
          <p className="text-xs text-gray-400 mt-3 text-center flex items-center justify-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            Your conversation is private and secure • Press Enter to send
          </p>
        </div>
      </div>
    </div>
  );
}
