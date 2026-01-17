/**
 * Tests for API hooks with 30s polling
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useEscalations, useEscalationStats, useReviewSubmission } from '@/hooks/useApi';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('useEscalations', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockFetch.mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should fetch escalations on mount', async () => {
    const mockData = {
      escalations: [
        { id: 'esc-1', reason: 'low_confidence', priority: 'medium', status: 'pending', timestamp: '2025-01-17T10:00:00Z' },
        { id: 'esc-2', reason: 'critical_finding', priority: 'high', status: 'pending', timestamp: '2025-01-17T10:05:00Z' },
      ],
      total: 2,
      timestamp: '2025-01-17T10:10:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useEscalations({}, false));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.escalations).toHaveLength(2);
    expect(result.current.total).toBe(2);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('should poll every 30 seconds when enabled', async () => {
    const mockData = {
      escalations: [],
      total: 0,
      timestamp: new Date().toISOString(),
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useEscalations({}, true));

    // Initial fetch
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    // Advance 30 seconds
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    // Advance another 30 seconds
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
  });

  it('should not poll when disabled', async () => {
    const mockData = {
      escalations: [],
      total: 0,
      timestamp: new Date().toISOString(),
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useEscalations({}, false));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    // Advance 60 seconds
    act(() => {
      jest.advanceTimersByTime(60000);
    });

    // Should still only have 1 call
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('should support filtering by priority', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ escalations: [], total: 0 }),
    });

    renderHook(() => useEscalations({ priority: 'high' }, false));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('priority=high'),
        expect.any(Object)
      );
    });
  });

  it('should handle errors gracefully', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useEscalations({}, false));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBeTruthy();
    expect(result.current.escalations).toHaveLength(0);
  });

  it('should allow manual refresh', async () => {
    const mockData = {
      escalations: [],
      total: 0,
      timestamp: new Date().toISOString(),
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useEscalations({}, false));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    // Manual refresh
    act(() => {
      result.current.refresh();
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  it('should track lastUpdated timestamp', async () => {
    const mockData = {
      escalations: [],
      total: 0,
      timestamp: new Date().toISOString(),
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useEscalations({}, false));

    await waitFor(() => {
      expect(result.current.lastUpdated).toBeTruthy();
    });

    expect(result.current.lastUpdated).toBeInstanceOf(Date);
  });
});

describe('useEscalationStats', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockFetch.mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should fetch stats on mount', async () => {
    const mockStats = {
      total_pending: 5,
      total_in_review: 2,
      total_approved_today: 10,
      total_rejected_today: 1,
      average_review_time_ms: 45000,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockStats),
    });

    const { result } = renderHook(() => useEscalationStats(false));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.stats).toEqual(mockStats);
  });

  it('should poll stats every 30 seconds', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ total_pending: 0 }),
    });

    renderHook(() => useEscalationStats(true));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    act(() => {
      jest.advanceTimersByTime(30000);
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });
});

describe('useReviewSubmission', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('should submit review successfully', async () => {
    const mockResponse = {
      id: 'esc-1',
      status: 'approved',
      reviewed_by: 'dr.smith',
      reviewed_at: '2025-01-17T10:00:00Z',
      timestamp: '2025-01-17T09:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const { result } = renderHook(() => useReviewSubmission());

    let reviewResult: any;
    await act(async () => {
      reviewResult = await result.current.submitReview('esc-1', {
        escalation_id: 'esc-1',
        decision: 'approve',
        notes: 'Approved with notes',
      });
    });

    expect(reviewResult).toBeTruthy();
    expect(reviewResult.status).toBe('approved');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/escalations/esc-1/review'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('approve'),
      })
    );
  });

  it('should handle review submission errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: () => Promise.resolve({ detail: 'Escalation not found' }),
    });

    const { result } = renderHook(() => useReviewSubmission());

    await act(async () => {
      await result.current.submitReview('non-existent', {
        escalation_id: 'non-existent',
        decision: 'approve',
        notes: 'Test',
      });
    });

    expect(result.current.error).toBeTruthy();
  });
});
