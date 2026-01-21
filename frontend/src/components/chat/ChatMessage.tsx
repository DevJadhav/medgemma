'use client';

import React from 'react';
import { Badge } from '@/components/ui/Badge';
import type { ChatMessage as ChatMessageType, TriageLevel, TRIAGE_LABELS } from '@/types/api';

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

export interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const formattedTime = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 message-enter`}>
      {/* Avatar for assistant */}
      {!isUser && (
        <div className="flex-shrink-0 mr-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
            <span className="text-sm">🏥</span>
          </div>
        </div>
      )}
      
      <div
        data-testid={isUser ? 'user-message' : 'assistant-message'}
        className={`
          max-w-[80%] rounded-2xl px-5 py-4 shadow-sm
          ${isUser 
            ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-br-md' 
            : 'bg-white text-gray-800 border border-gray-100 rounded-bl-md'
          }
        `}
      >
        {/* Triage badge for assistant messages */}
        {!isUser && message.triageLevel && (
          <div className="mb-3">
            <Badge
              data-testid="triage-badge"
              variant={triageBadgeVariants[message.triageLevel]}
              pulse={message.triageLevel === 'EMERGENCY'}
              size="sm"
            >
              {triageLabels[message.triageLevel]}
            </Badge>
          </div>
        )}

        {/* Message content */}
        <p className={`text-sm leading-relaxed whitespace-pre-wrap ${isUser ? 'text-white' : 'text-gray-700'}`}>
          {message.content}
        </p>

        {/* Timestamp */}
        <p
          data-testid="message-timestamp"
          className={`text-xs mt-3 ${isUser ? 'text-primary-100' : 'text-gray-400'}`}
        >
          {formattedTime}
        </p>
      </div>

      {/* Avatar for user */}
      {isUser && (
        <div className="flex-shrink-0 ml-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-gray-600 to-gray-700 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}
