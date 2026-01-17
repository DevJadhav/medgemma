import React from 'react';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'emergency' | 'urgent' | 'success' | 'info' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
  className?: string;
}

const variantStyles = {
  default: 'bg-gray-100 text-gray-800',
  emergency: 'bg-emergency-100 text-emergency-800',
  urgent: 'bg-urgent-100 text-urgent-800',
  success: 'bg-success-100 text-success-800',
  info: 'bg-primary-100 text-primary-800',
  warning: 'bg-yellow-100 text-yellow-800',
};

const sizeStyles = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-0.5 text-sm',
  lg: 'px-3 py-1 text-base',
};

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  pulse = false,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center font-medium rounded-full
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${pulse ? 'animate-pulse' : ''}
        ${className}
      `}
    >
      {children}
    </span>
  );
}
