'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/Button';

export interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  loading?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  loading = false,
  placeholder = 'Type your message here...',
}: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (trimmedInput && !disabled && !loading) {
      onSend(trimmedInput);
      setInput('');
    }
  };

  return (
    <form role="form" onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={placeholder}
        disabled={disabled || loading}
        className="
          flex-1 px-4 py-3 rounded-lg border border-gray-300
          focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
          disabled:bg-gray-100 disabled:cursor-not-allowed
          text-gray-900 placeholder-gray-500
        "
      />
      <Button
        type="submit"
        disabled={disabled || loading || !input.trim()}
        loading={loading}
        aria-label="Send"
      >
        {loading ? 'Sending...' : 'Send'}
      </Button>
    </form>
  );
}
