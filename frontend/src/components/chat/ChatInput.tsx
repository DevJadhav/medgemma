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
    <form role="form" onSubmit={handleSubmit} className="flex gap-3">
      <div className="flex-1 relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || loading}
          className="
            w-full px-5 py-4 rounded-2xl border-2 border-gray-200 bg-white
            text-gray-800 placeholder-gray-400
            transition-all duration-200 ease-out
            focus:border-primary-500 focus:ring-4 focus:ring-primary-100 focus:outline-none
            hover:border-gray-300
            disabled:bg-gray-50 disabled:cursor-not-allowed disabled:border-gray-200
            text-sm
          "
        />
        {input.length > 0 && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-gray-400">
            {input.length}/500
          </span>
        )}
      </div>
      <Button
        type="submit"
        disabled={disabled || loading || !input.trim()}
        loading={loading}
        aria-label="Send"
        className="px-6"
      >
        {loading ? (
          'Sending...'
        ) : (
          <span className="flex items-center gap-2">
            Send
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </span>
        )}
      </Button>
    </form>
  );
}
