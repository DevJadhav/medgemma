'use client';

import React, { useRef, useEffect, useState } from 'react';
import { useChat } from '@/hooks/useChat';
import { useRAG, type RAGDocument } from '@/hooks/useApi';
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

  // RAG integration
  const { query: ragQuery, loading: ragLoading } = useRAG();
  const [ragEnabled, setRagEnabled] = useState(false);
  const [ragSources, setRagSources] = useState<RAGDocument[]>([]);
  const [showSources, setShowSources] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Enhanced send with RAG
  const handleSendWithRAG = async (message: string) => {
    if (ragEnabled) {
      // First, get RAG context
      const ragResult = await ragQuery(message, { top_k: 3, include_sources: true });
      if (ragResult?.sources) {
        setRagSources(ragResult.sources);
      }
    } else {
      setRagSources([]);
    }
    // Then send the message
    await sendMessage(message);
  };

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
        <div className="flex items-center gap-3">
          {/* RAG Toggle */}
          <button
            onClick={() => setRagEnabled(!ragEnabled)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              ragEnabled
                ? 'bg-primary-100 text-primary-700 border border-primary-200'
                : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
            }`}
            title="Toggle RAG-enhanced responses"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            RAG {ragEnabled ? 'ON' : 'OFF'}
          </button>
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

      {/* RAG Sources Panel */}
      {ragEnabled && ragSources.length > 0 && (
        <div className="px-5 py-3 border-t border-gray-100 bg-gray-50">
          <button
            onClick={() => setShowSources(!showSources)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showSources ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="font-medium">Sources ({ragSources.length})</span>
            <Badge variant="info" size="sm">RAG</Badge>
          </button>
          {showSources && (
            <div className="mt-3 space-y-2 max-h-48 overflow-y-auto">
              {ragSources.map((source, idx) => (
                <div
                  key={source.id || idx}
                  className="p-3 bg-white rounded-lg border border-gray-200 text-sm"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-700">Source {idx + 1}</span>
                    {source.relevance_score && (
                      <Badge variant="success" size="sm">
                        {(source.relevance_score * 100).toFixed(0)}% match
                      </Badge>
                    )}
                  </div>
                  <p className="text-gray-600 line-clamp-2">{source.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input area */}
      <div className="p-5 border-t border-gray-100 bg-white">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSend={handleSendWithRAG}
            loading={loading || ragLoading}
            disabled={false}
            placeholder={ragEnabled
              ? "Ask a question (RAG-enhanced with medical knowledge)..."
              : "Describe your symptoms or ask a health question..."
            }
          />
          <p className="text-xs text-gray-400 mt-3 text-center flex items-center justify-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            Your conversation is private and secure
            {ragEnabled && ' • RAG-enhanced'}
            {' • Press Enter to send'}
          </p>
        </div>
      </div>
    </div>
  );
}
