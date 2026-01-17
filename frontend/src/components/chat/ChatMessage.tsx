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
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div
        data-testid={isUser ? 'user-message' : 'assistant-message'}
        className={`
          max-w-[80%] rounded-lg px-4 py-3
          ${isUser ? 'bg-primary-100 text-primary-900' : 'bg-gray-100 text-gray-900'}
        `}
      >
        {/* Triage badge for assistant messages */}
        {!isUser && message.triageLevel && (
          <div className="mb-2">
            <Badge
              data-testid="triage-badge"
              variant={triageBadgeVariants[message.triageLevel]}
              pulse={message.triageLevel === 'EMERGENCY'}
            >
              {triageLabels[message.triageLevel]}
            </Badge>
          </div>
        )}

        {/* Message content */}
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>

        {/* Timestamp */}
        <p
          data-testid="message-timestamp"
          className={`text-xs mt-2 ${isUser ? 'text-primary-600' : 'text-gray-500'}`}
        >
          {formattedTime}
        </p>
      </div>
    </div>
  );
}
