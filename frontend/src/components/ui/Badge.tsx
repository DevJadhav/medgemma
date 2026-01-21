import React from 'react';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'emergency' | 'urgent' | 'success' | 'info' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
  className?: string;
}

const variantStyles = {
  default: 'bg-gray-100 text-gray-700 border-gray-200',
  emergency: 'bg-gradient-to-r from-red-500 to-red-600 text-white border-transparent shadow-sm shadow-red-500/25',
  urgent: 'bg-gradient-to-r from-orange-400 to-orange-500 text-white border-transparent shadow-sm shadow-orange-500/25',
  success: 'bg-gradient-to-r from-emerald-400 to-emerald-500 text-white border-transparent shadow-sm shadow-emerald-500/25',
  info: 'bg-gradient-to-r from-blue-400 to-blue-500 text-white border-transparent shadow-sm shadow-blue-500/25',
  warning: 'bg-gradient-to-r from-amber-400 to-amber-500 text-white border-transparent shadow-sm shadow-amber-500/25',
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
