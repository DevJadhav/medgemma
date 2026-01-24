import React from 'react';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'emergency' | 'urgent' | 'success' | 'info' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
  className?: string;
}

const variantStyles = {
  default: 'bg-secondary text-secondary-foreground border-transparent hover:bg-secondary/80',
  emergency: 'bg-emergency-100 text-emergency-700 border-emergency-200',
  urgent: 'bg-urgent-100 text-urgent-700 border-urgent-200',
  success: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  info: 'bg-blue-100 text-blue-700 border-blue-200',
  warning: 'bg-amber-100 text-amber-700 border-amber-200',
};

const sizeStyles = {
  sm: 'px-2.5 py-0.5 text-xs',
  md: 'px-3 py-1 text-xs',
  lg: 'px-4 py-1.5 text-sm',
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
        inline-flex items-center font-semibold rounded-full border
        transition-all duration-200
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${pulse ? 'animate-pulse' : ''}
        ${className}
      `}
    >
      {pulse && (
        <span className="w-1.5 h-1.5 rounded-full bg-current mr-1.5 animate-ping opacity-75"></span>
      )}
      {children}
    </span>
  );
}
