import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PatientChat } from '@/components/chat/PatientChat';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatInput } from '@/components/chat/ChatInput';
import { MedicalDisclaimer } from '@/components/chat/MedicalDisclaimer';

// Mock the hooks
jest.mock('@/hooks/useChat', () => ({
  useChat: () => ({
    messages: [],
    sessionId: null,
    currentTriageLevel: null,
    isEmergency: false,
    loading: false,
    error: null,
    sendMessage: jest.fn(),
    clearChat: jest.fn(),
  }),
}));

// =============================================================================
// ChatMessage Tests
// =============================================================================

describe('ChatMessage', () => {
  it('renders user message correctly', () => {
    render(
      <ChatMessage
        message={{
          id: '1',
          role: 'user',
          content: 'I have a headache',
          timestamp: '2024-01-15T10:00:00Z',
        }}
      />
    );
    expect(screen.getByText('I have a headache')).toBeInTheDocument();
    expect(screen.getByTestId('user-message')).toHaveClass('bg-primary-100');
  });

  it('renders assistant message correctly', () => {
    render(
      <ChatMessage
        message={{
          id: '2',
          role: 'assistant',
          content: 'Can you describe the pain?',
          timestamp: '2024-01-15T10:01:00Z',
        }}
      />
    );
    expect(screen.getByText('Can you describe the pain?')).toBeInTheDocument();
    expect(screen.getByTestId('assistant-message')).toHaveClass('bg-gray-100');
  });

  it('displays triage level badge when present', () => {
    render(
      <ChatMessage
        message={{
          id: '3',
          role: 'assistant',
          content: 'This needs attention',
          timestamp: '2024-01-15T10:02:00Z',
          triageLevel: 'URGENT',
        }}
      />
    );
    expect(screen.getByText('Urgent')).toBeInTheDocument();
  });

  it('displays emergency styling for emergency triage', () => {
    render(
      <ChatMessage
        message={{
          id: '4',
          role: 'assistant',
          content: 'Call 911',
          timestamp: '2024-01-15T10:03:00Z',
          triageLevel: 'EMERGENCY',
        }}
      />
    );
    expect(screen.getByText('Emergency')).toBeInTheDocument();
    expect(screen.getByTestId('triage-badge')).toHaveClass('animate-pulse');
  });

  it('formats timestamp correctly', () => {
    render(
      <ChatMessage
        message={{
          id: '5',
          role: 'user',
          content: 'Test',
          timestamp: '2024-01-15T10:30:00Z',
        }}
      />
    );
    // Should show formatted time
    expect(screen.getByTestId('message-timestamp')).toBeInTheDocument();
  });
});

// =============================================================================
// ChatInput Tests
// =============================================================================

describe('ChatInput', () => {
  it('renders input field and send button', () => {
    render(<ChatInput onSend={jest.fn()} />);
    expect(screen.getByPlaceholderText(/type your message/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('calls onSend when form is submitted', async () => {
    const handleSend = jest.fn();
    render(<ChatInput onSend={handleSend} />);

    const input = screen.getByPlaceholderText(/type your message/i);
    await userEvent.type(input, 'Hello');
    fireEvent.submit(screen.getByRole('form'));

    expect(handleSend).toHaveBeenCalledWith('Hello');
  });

  it('clears input after sending', async () => {
    const handleSend = jest.fn();
    render(<ChatInput onSend={handleSend} />);

    const input = screen.getByPlaceholderText(/type your message/i);
    await userEvent.type(input, 'Hello');
    fireEvent.submit(screen.getByRole('form'));

    expect(input).toHaveValue('');
  });

  it('disables input when disabled prop is true', () => {
    render(<ChatInput onSend={jest.fn()} disabled />);
    expect(screen.getByPlaceholderText(/type your message/i)).toBeDisabled();
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();
  });

  it('shows loading state when loading', () => {
    render(<ChatInput onSend={jest.fn()} loading />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('does not submit empty messages', async () => {
    const handleSend = jest.fn();
    render(<ChatInput onSend={handleSend} />);

    fireEvent.submit(screen.getByRole('form'));
    expect(handleSend).not.toHaveBeenCalled();
  });
});

// =============================================================================
// MedicalDisclaimer Tests
// =============================================================================

describe('MedicalDisclaimer', () => {
  it('renders disclaimer text', () => {
    render(<MedicalDisclaimer />);
    expect(screen.getByText(/not a substitute for professional medical advice/i)).toBeInTheDocument();
  });

  it('shows emergency instructions', () => {
    render(<MedicalDisclaimer />);
    expect(screen.getByText(/call 112/i)).toBeInTheDocument();
  });

  it('can be collapsed', async () => {
    render(<MedicalDisclaimer collapsible />);
    const toggleButton = screen.getByRole('button', { name: /toggle disclaimer/i });
    
    // Initially expanded
    expect(screen.getByText(/not a substitute/i)).toBeVisible();
    
    // Click to collapse
    await userEvent.click(toggleButton);
    await waitFor(() => {
      expect(screen.queryByText(/not a substitute/i)).not.toBeVisible();
    });
  });
});

// =============================================================================
// PatientChat Integration Tests
// =============================================================================

describe('PatientChat', () => {
  it('renders chat interface', () => {
    render(<PatientChat />);
    expect(screen.getByTestId('patient-chat')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/type your message/i)).toBeInTheDocument();
  });

  it('shows medical disclaimer', () => {
    render(<PatientChat />);
    expect(screen.getByText(/not a substitute for professional medical advice/i)).toBeInTheDocument();
  });

  it('displays welcome message initially', () => {
    render(<PatientChat />);
    expect(screen.getByText(/hello! how can I help you today/i)).toBeInTheDocument();
  });

  it('shows emergency alert when triage level is EMERGENCY', () => {
    // Override the mock for this test
    jest.spyOn(require('@/hooks/useChat'), 'useChat').mockReturnValue({
      messages: [
        {
          id: '1',
          role: 'assistant',
          content: 'Call 112 immediately',
          timestamp: '2024-01-15T10:00:00Z',
          triageLevel: 'EMERGENCY',
        },
      ],
      sessionId: 'test-session',
      currentTriageLevel: 'EMERGENCY',
      isEmergency: true,
      loading: false,
      error: null,
      sendMessage: jest.fn(),
      clearChat: jest.fn(),
    });

    render(<PatientChat />);
    expect(screen.getByRole('alert')).toHaveClass('bg-emergency-600');
  });

  it('displays error message when there is an error', () => {
    jest.spyOn(require('@/hooks/useChat'), 'useChat').mockReturnValue({
      messages: [],
      sessionId: null,
      currentTriageLevel: null,
      isEmergency: false,
      loading: false,
      error: 'Failed to connect',
      sendMessage: jest.fn(),
      clearChat: jest.fn(),
    });

    render(<PatientChat />);
    expect(screen.getByText(/failed to connect/i)).toBeInTheDocument();
  });

  it('shows triage indicator based on current level', () => {
    jest.spyOn(require('@/hooks/useChat'), 'useChat').mockReturnValue({
      messages: [],
      sessionId: 'test',
      currentTriageLevel: 'URGENT',
      isEmergency: false,
      loading: false,
      error: null,
      sendMessage: jest.fn(),
      clearChat: jest.fn(),
    });

    render(<PatientChat />);
    expect(screen.getByTestId('triage-indicator')).toHaveTextContent('Urgent');
  });
});
