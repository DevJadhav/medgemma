import React from 'react';
import { cn } from '@/lib/utils';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'outlined' | 'glass';
  hover?: boolean;
}

const variantStyles = {
  default: 'bg-card text-card-foreground border border-border shadow-sm',
  elevated: 'bg-card text-card-foreground border-none shadow-md',
  outlined: 'bg-transparent border border-border shadow-none',
  glass: 'glass text-foreground',
};

export function Card({ children, className, variant = 'default', hover = false, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl",
        variantStyles[variant],
        hover && "transition-transform duration-300 hover:-translate-y-1 hover:shadow-lg",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("p-6 flex flex-col gap-1.5", className)} {...props}>
      {children}
    </div>
  );
}

export function CardContent({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("p-6 pt-0", className)} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("p-6 pt-0 flex items-center", className)} {...props}>
      {children}
    </div>
  );
}
