import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardContent, CardFooter } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Alert, EmergencyAlert } from '@/components/ui/Alert';

// =============================================================================
// Button Tests
// =============================================================================

describe('Button', () => {
  it('renders with default variant', () => {
    render(<Button>Click me</Button>);
    const button = screen.getByRole('button', { name: /click me/i });
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass('bg-primary-600');
  });

  it('renders with secondary variant', () => {
    render(<Button variant="secondary">Secondary</Button>);
    const button = screen.getByRole('button', { name: /secondary/i });
    expect(button).toHaveClass('bg-gray-100');
  });

  it('renders with danger variant', () => {
    render(<Button variant="danger">Delete</Button>);
    const button = screen.getByRole('button', { name: /delete/i });
    expect(button).toHaveClass('bg-emergency-600');
  });

  it('renders with outline variant', () => {
    render(<Button variant="outline">Outline</Button>);
    const button = screen.getByRole('button', { name: /outline/i });
    expect(button).toHaveClass('border-gray-300');
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('can be disabled', () => {
    const handleClick = jest.fn();
    render(<Button disabled onClick={handleClick}>Disabled</Button>);
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('shows loading state', () => {
    render(<Button loading>Loading</Button>);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('supports different sizes', () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    expect(screen.getByRole('button')).toHaveClass('px-3', 'py-1.5');

    rerender(<Button size="lg">Large</Button>);
    expect(screen.getByRole('button')).toHaveClass('px-6', 'py-3');
  });
});

// =============================================================================
// Card Tests
// =============================================================================

describe('Card', () => {
  it('renders card with content', () => {
    render(
      <Card>
        <CardContent>Card content</CardContent>
      </Card>
    );
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('renders card with header, content, and footer', () => {
    render(
      <Card>
        <CardHeader>Header</CardHeader>
        <CardContent>Content</CardContent>
        <CardFooter>Footer</CardFooter>
      </Card>
    );
    expect(screen.getByText('Header')).toBeInTheDocument();
    expect(screen.getByText('Content')).toBeInTheDocument();
    expect(screen.getByText('Footer')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<Card className="custom-class">Content</Card>);
    expect(screen.getByText('Content').parentElement).toHaveClass('custom-class');
  });
});

// =============================================================================
// Badge Tests
// =============================================================================

describe('Badge', () => {
  it('renders with default variant', () => {
    render(<Badge>Default</Badge>);
    expect(screen.getByText('Default')).toHaveClass('bg-gray-100');
  });

  it('renders triage badges correctly', () => {
    const { rerender } = render(<Badge variant="emergency">Emergency</Badge>);
    expect(screen.getByText('Emergency')).toHaveClass('bg-emergency-100');

    rerender(<Badge variant="urgent">Urgent</Badge>);
    expect(screen.getByText('Urgent')).toHaveClass('bg-urgent-100');

    rerender(<Badge variant="success">Success</Badge>);
    expect(screen.getByText('Success')).toHaveClass('bg-success-100');
  });

  it('renders with pulse animation for emergency', () => {
    render(<Badge variant="emergency" pulse>Critical</Badge>);
    expect(screen.getByText('Critical')).toHaveClass('animate-pulse');
  });
});

// =============================================================================
// Alert Tests
// =============================================================================

describe('Alert', () => {
  it('renders info alert by default', () => {
    render(<Alert>Information message</Alert>);
    const alert = screen.getByRole('alert');
    expect(alert).toHaveClass('bg-primary-50');
    expect(screen.getByText('Information message')).toBeInTheDocument();
  });

  it('renders warning alert', () => {
    render(<Alert variant="warning">Warning message</Alert>);
    expect(screen.getByRole('alert')).toHaveClass('bg-urgent-50');
  });

  it('renders error alert', () => {
    render(<Alert variant="error">Error message</Alert>);
    expect(screen.getByRole('alert')).toHaveClass('bg-emergency-50');
  });

  it('renders success alert', () => {
    render(<Alert variant="success">Success message</Alert>);
    expect(screen.getByRole('alert')).toHaveClass('bg-success-50');
  });

  it('shows title when provided', () => {
    render(<Alert title="Alert Title">Message</Alert>);
    expect(screen.getByText('Alert Title')).toBeInTheDocument();
  });

  it('can be dismissed', () => {
    const onDismiss = jest.fn();
    render(<Alert dismissible onDismiss={onDismiss}>Dismissible</Alert>);
    
    const dismissButton = screen.getByRole('button', { name: /dismiss/i });
    fireEvent.click(dismissButton);
    expect(onDismiss).toHaveBeenCalled();
  });
});

describe('EmergencyAlert', () => {
  it('renders emergency alert with correct styling', () => {
    render(<EmergencyAlert>Call 112 immediately</EmergencyAlert>);
    const alert = screen.getByRole('alert');
    expect(alert).toHaveClass('bg-emergency-600');
    expect(screen.getByText('Call 112 immediately')).toBeInTheDocument();
  });

  it('shows emergency icon', () => {
    render(<EmergencyAlert>Emergency!</EmergencyAlert>);
    expect(screen.getByTestId('emergency-icon')).toBeInTheDocument();
  });

  it('has pulsing animation', () => {
    render(<EmergencyAlert>Emergency!</EmergencyAlert>);
    expect(screen.getByRole('alert')).toHaveClass('animate-pulse');
  });
});
