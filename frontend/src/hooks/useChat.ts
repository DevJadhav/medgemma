'use client';

import { useState, useCallback } from 'react';
import { useCommunication } from './useApi';
import type { ChatMessage, TriageLevel } from '@/types/api';

export interface ChatState {
  messages: ChatMessage[];
  sessionId: string | null;
  currentTriageLevel: TriageLevel | null;
  isEmergency: boolean;
}

export function useChat(patientId: string = 'anonymous') {
  const [state, setState] = useState<ChatState>({
    messages: [],
    sessionId: null,
    currentTriageLevel: null,
    isEmergency: false,
  });

  const { sendMessage: apiSendMessage, loading, error } = useCommunication();

  const sendMessage = useCallback(async (content: string) => {
    // Add user message immediately
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
    }));

    // Send to API
    const response = await apiSendMessage({
      message: content,
      patient_id: patientId,
      session_id: state.sessionId || undefined,
    });

    if (response) {
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        triageLevel: response.triage_level,
      };

      const isEmergency = response.triage_level === 'EMERGENCY';

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        sessionId: response.session_id,
        currentTriageLevel: response.triage_level,
        isEmergency,
      }));

      return response;
    }

    return null;
  }, [apiSendMessage, patientId, state.sessionId]);

  const clearChat = useCallback(() => {
    setState({
      messages: [],
      sessionId: null,
      currentTriageLevel: null,
      isEmergency: false,
    });
  }, []);

  return {
    messages: state.messages,
    sessionId: state.sessionId,
    currentTriageLevel: state.currentTriageLevel,
    isEmergency: state.isEmergency,
    loading,
    error,
    sendMessage,
    clearChat,
  };
}
