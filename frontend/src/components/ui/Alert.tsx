import React from 'react';

export interface AlertProps {
  children: React.ReactNode;
  variant?: 'info' | 'warning' | 'error' | 'success';
  title?: string;
  dismissible?: boolean;
  onDismiss?: () => void;
  className?: string;
}

const variantStyles = {
  info: 'bg-blue-50 border-l-4 border-blue-500 text-blue-800',
  warning: 'bg-amber-50 border-l-4 border-amber-500 text-amber-800',
  error: 'bg-red-50 border-l-4 border-red-500 text-red-800',
  success: 'bg-emerald-50 border-l-4 border-emerald-500 text-emerald-800',
};

const iconPaths = {
  info: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  warning: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  error: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z',
  success: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
};

const iconColors = {
  info: 'text-blue-500',
  warning: 'text-amber-500',
  error: 'text-red-500',
  success: 'text-emerald-500',
};

export function Alert({
  children,
  variant = 'info',
  title,
  dismissible = false,
  onDismiss,
  className = '',
}: AlertProps) {
  return (
    <div
      role="alert"
      className={`
        flex items-start p-4 rounded-xl
        ${variantStyles[variant]}
        ${className}
      `}
    >
      <svg
        className={`w-5 h-5 mr-3 flex-shrink-0 mt-0.5 ${iconColors[variant]}`}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d={iconPaths[variant]}
        />
      </svg>
      <div className="flex-1">
        {title && <h4 className="font-semibold mb-1">{title}</h4>}
        <div className="text-sm leading-relaxed">{children}</div>
      </div>
      {dismissible && (
        <button
          onClick={onDismiss}
          className="ml-3 flex-shrink-0 -mr-1 -mt-1 p-1.5 rounded-lg hover:bg-black/5 transition-colors"
          aria-label="Dismiss"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      )}
    </div>
  );
}

export interface EmergencyAlertProps {
  children: React.ReactNode;
  className?: string;
}

export function EmergencyAlert({ children, className = '' }: EmergencyAlertProps) {
  return (
    <div
      role="alert"
      className={`
        flex items-center p-5 rounded-2xl
        bg-gradient-to-r from-red-500 to-red-600 text-white
        shadow-lg shadow-red-500/25
        emergency-pulse
        ${className}
      `}
    >
      <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center mr-4 flex-shrink-0">
        <svg
          data-testid="emergency-icon"
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </div>
      <div>
        <p className="font-bold text-lg mb-1">Emergency Alert</p>
        <p className="text-white/90 text-sm">{children}</p>
      </div>
    </div>
  );
}
