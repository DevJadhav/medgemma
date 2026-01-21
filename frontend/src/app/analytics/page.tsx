'use client';

import React, { useMemo } from 'react';
import { Navigation } from '@/components/Navigation';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { useEscalationStats, useEscalations, useSystemMetrics } from '@/hooks/useApi';
import { Loader2, CheckCircle, XCircle, Clock, Cpu, Database, Server, Activity } from 'lucide-react';
import { format, subDays, isSameDay, parseISO } from 'date-fns';

function MetricCard({
  title,
  value,
  subtitle,
  variant = 'default'
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  variant?: 'default' | 'success' | 'warning' | 'error';
}) {
  const colors = {
    default: 'text-gray-900',
    success: 'text-success-600',
    warning: 'text-urgent-600',
    error: 'text-emergency-600',
  };

  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-sm text-gray-500 mb-1">{title}</p>
        <p className={`text-2xl font-bold ${colors[variant]}`}>{value}</p>
        {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

function DailyActivityChart({ data }: { data: { date: string; requests: number; escalations: number }[] }) {
  if (data.length === 0) return <div className="h-40 flex items-center justify-center text-muted-foreground text-sm">No recent activity</div>;

  const maxVal = Math.max(...data.map(d => Math.max(d.requests, d.escalations)), 5); // Minimum scale of 5

  return (
    <div className="flex items-end justify-between h-40 gap-2">
      {data.map((day, index) => (
        <div key={index} className="flex-1 flex flex-col items-center group relative">
          <div className="w-full flex flex-col items-center justify-end h-[120px] gap-1">
            {/* Tooltip */}
            <div className="absolute bottom-full mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 text-white text-[10px] p-1 rounded pointer-events-none whitespace-nowrap z-10">
              {day.date}: {day.escalations} escalations
            </div>

            {/* Escalations Bar (solid) */}
            <div
              className="w-full max-w-[20px] bg-urgent-500 rounded-t transition-all duration-500 ease-in-out"
              style={{ height: `${Math.max((day.escalations / maxVal) * 100, 4)}%` }} // Min height 4%
            />
          </div>
          <span className="text-[10px] text-gray-500 mt-2 truncate w-full text-center">{day.date}</span>
        </div>
      ))}
    </div>
  );
}

export default function AnalyticsPage() {
  const { stats, loading: statsLoading } = useEscalationStats(false);
  // Fetch a larger list of escalations to compute client-side analytics
  const { escalations, loading: listLoading } = useEscalations({ limit: 100 });
  // Real-time system metrics with 10s polling
  const { metrics: systemMetrics, loading: metricsLoading } = useSystemMetrics(true);

  const analytics = useMemo(() => {
    if (!escalations) return null;

    // 1. Critical Findings Today
    const today = new Date();
    const criticalToday = escalations.filter(e =>
      e.reason === 'critical_finding' &&
      isSameDay(parseISO(e.timestamp as string), today)
    ).length;

    // 2. Agent Performance (based on escalation source)
    const agentCounts: Record<string, number> = {};
    const agentConfidences: Record<string, number[]> = {};

    escalations.forEach(e => {
      const agent = e.agent_type || 'Unknown';
      agentCounts[agent] = (agentCounts[agent] || 0) + 1;
      if (e.confidence_score) {
        if (!agentConfidences[agent]) agentConfidences[agent] = [];
        agentConfidences[agent].push(e.confidence_score);
      }
    });

    const agentMetrics = Object.entries(agentCounts).map(([agent, count]) => {
      const scores = agentConfidences[agent] || [];
      const avgConf = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
      return {
        agent,
        count,
        avgConfidence: (avgConf * 100).toFixed(1)
      };
    });

    // 3. Daily Activity (Last 7 Days)
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const d = subDays(today, 6 - i);
      return {
        date: format(d, 'MMM dd'),
        iso: format(d, 'yyyy-MM-dd'),
        requests: 0, // We cannot track total requests from here, only escalations
        escalations: 0
      };
    });

    escalations.forEach(e => {
      const eDate = format(parseISO(e.timestamp as string), 'yyyy-MM-dd');
      const dayStat = last7Days.find(d => d.iso === eDate);
      if (dayStat) {
        dayStat.escalations += 1;
      }
    });

    return {
      criticalToday,
      agentMetrics,
      dailyStats: last7Days
    };
  }, [escalations]);


  const loading = statsLoading || listLoading || metricsLoading;

  // Format uptime
  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 pb-12">
      <Navigation pendingReviews={stats?.pending_reviews || 0} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <Badge variant="info" className="mb-2 bg-blue-50 text-blue-700 border-blue-200">
            Live Data
          </Badge>
          <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Real-time system performance and clinical escalation metrics.
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <MetricCard
            title="Pending Reviews"
            value={stats?.total_pending || 0}
            subtitle={stats?.total_in_review ? `${stats.total_in_review} in review` : 'No active reviews'}
            variant={stats?.total_pending && stats.total_pending > 5 ? 'warning' : 'default'}
          />
          <MetricCard
            title="Avg Review Time"
            value={stats?.average_review_time_ms ? `${(stats.average_review_time_ms / 1000).toFixed(1)}s` : '0s'}
            subtitle="Target: <30s"
            variant="success"
          />
          <MetricCard
            title="Decisions Today"
            value={(stats?.total_approved_today || 0) + (stats?.total_rejected_today || 0)}
            subtitle={`${stats?.total_approved_today || 0} approved`}
            variant="default"
          />
          <MetricCard
            title="Critical Findings"
            value={analytics?.criticalToday || 0}
            subtitle="Flagged today"
            variant={(analytics?.criticalToday || 0) > 0 ? 'error' : 'success'}
          />
        </div>

        {/* Charts Row */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Daily Activity Chart */}
          <Card className="overflow-hidden">
            <CardHeader className="border-b bg-gray-50/50">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Weekly Escalation Volume</h2>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              <DailyActivityChart data={analytics?.dailyStats || []} />
            </CardContent>
          </Card>

          {/* Agent Performance */}
          <Card className="overflow-hidden">
            <CardHeader className="border-b bg-gray-50/50">
              <h2 className="text-lg font-semibold text-gray-900">Agent Escalation Sources</h2>
            </CardHeader>
            <CardContent className="p-0">
              {analytics?.agentMetrics && analytics.agentMetrics.length > 0 ? (
                <div className="divide-y divide-gray-100">
                  {analytics.agentMetrics.map((am, index) => (
                    <div key={index} className="flex items-center justify-between p-4 hover:bg-gray-50 transition-colors">
                      <div>
                        <p className="font-medium text-gray-900 capitalize">{am.agent.replace('_', ' ')} Agent</p>
                        <p className="text-xs text-muted-foreground">{am.count} escalations</p>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm font-bold">{am.avgConfidence}%</p>
                          <p className="text-[10px] text-gray-400 uppercase tracking-wider">Avg Conf</p>
                        </div>
                        <Badge variant={parseFloat(am.avgConfidence) >= 90 ? 'success' : 'warning'}>
                          {parseFloat(am.avgConfidence) >= 90 ? 'High' : 'Review'}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center text-muted-foreground">
                  No agent data available yet.
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* System Health & Performance */}
        <Card className="overflow-hidden mb-8">
          <CardHeader className="border-b bg-gray-50/50">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">System Health & Performance</h2>
              <Badge variant={systemMetrics?.model_status === 'online' ? 'success' : 'warning'}>
                {systemMetrics?.model_status === 'online' ? 'All Systems Operational' : 'Degraded'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {/* Model Status */}
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${systemMetrics?.model_status === 'online' ? 'bg-success-100' : 'bg-gray-100'}`}>
                  <Cpu className={`w-5 h-5 ${systemMetrics?.model_status === 'online' ? 'text-success-600' : 'text-gray-400'}`} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">MedGemma 27B</p>
                  <div className="flex items-center gap-1">
                    {systemMetrics?.model_status === 'online' ? (
                      <CheckCircle className="w-4 h-4 text-success-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-gray-400" />
                    )}
                    <span className="text-sm font-medium capitalize">{systemMetrics?.model_status || 'unknown'}</span>
                  </div>
                </div>
              </div>

              {/* GPU Status */}
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${systemMetrics?.gpu_available ? 'bg-success-100' : 'bg-gray-100'}`}>
                  <Server className={`w-5 h-5 ${systemMetrics?.gpu_available ? 'text-success-600' : 'text-gray-400'}`} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">GPU</p>
                  <p className="text-sm font-medium">{systemMetrics?.gpu_name || 'Not Available'}</p>
                </div>
              </div>

              {/* Database */}
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${systemMetrics?.postgres_connected ? 'bg-success-100' : 'bg-emergency-100'}`}>
                  <Database className={`w-5 h-5 ${systemMetrics?.postgres_connected ? 'text-success-600' : 'text-emergency-600'}`} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">PostgreSQL</p>
                  <div className="flex items-center gap-1">
                    {systemMetrics?.postgres_connected ? (
                      <CheckCircle className="w-4 h-4 text-success-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-emergency-600" />
                    )}
                    <span className="text-sm font-medium">{systemMetrics?.postgres_connected ? 'Connected' : 'Disconnected'}</span>
                  </div>
                </div>
              </div>

              {/* Redis Cache */}
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${systemMetrics?.redis_connected ? 'bg-success-100' : 'bg-urgent-100'}`}>
                  <Activity className={`w-5 h-5 ${systemMetrics?.redis_connected ? 'text-success-600' : 'text-urgent-600'}`} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">Redis Cache</p>
                  <div className="flex items-center gap-1">
                    {systemMetrics?.redis_connected ? (
                      <CheckCircle className="w-4 h-4 text-success-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-urgent-600" />
                    )}
                    <span className="text-sm font-medium">{systemMetrics?.redis_connected ? 'Connected' : 'Disconnected'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics Row */}
            <div className="mt-6 pt-6 border-t border-gray-100 grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{formatUptime(systemMetrics?.uptime_seconds || 0)}</p>
                <p className="text-xs text-gray-500">Uptime</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{systemMetrics?.active_sessions || 0}</p>
                <p className="text-xs text-gray-500">Active Sessions</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{systemMetrics?.avg_response_time_ms?.toFixed(0) || 0}ms</p>
                <p className="text-xs text-gray-500">Avg Response</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{systemMetrics?.inference_queue_size || 0}</p>
                <p className="text-xs text-gray-500">Queue Size</p>
              </div>
              <div className="text-center">
                <p className={`text-2xl font-bold ${(systemMetrics?.recent_errors || 0) > 0 ? 'text-emergency-600' : 'text-success-600'}`}>
                  {systemMetrics?.recent_errors || 0}
                </p>
                <p className="text-xs text-gray-500">Recent Errors</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Note on Real Data */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex gap-3 text-sm text-blue-800">
          <span className="text-lg">ℹ️</span>
          <div>
            <span className="font-semibold">Data Integrity:</span> This dashboard now reflects live production data directly from the MedAI Compass escalation engine.
            Metrics are calculated in real-time based on active and historical case files. System metrics refresh every 10 seconds.
          </div>
        </div>
      </main>
    </div>
  );
}
