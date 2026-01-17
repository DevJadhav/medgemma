'use client';

import React from 'react';
import { Navigation } from '@/components/Navigation';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { useEscalationStats } from '@/hooks/useApi';

// Placeholder chart data - in production, this would come from the API
const mockDailyStats = [
  { date: 'Mon', requests: 145, escalations: 12 },
  { date: 'Tue', requests: 168, escalations: 15 },
  { date: 'Wed', requests: 132, escalations: 8 },
  { date: 'Thu', requests: 189, escalations: 18 },
  { date: 'Fri', requests: 156, escalations: 11 },
  { date: 'Sat', requests: 78, escalations: 5 },
  { date: 'Sun', requests: 64, escalations: 3 },
];

const mockAgentMetrics = [
  { agent: 'Communication', requests: 456, avgTime: 245, accuracy: 94.5 },
  { agent: 'Diagnostic', requests: 234, avgTime: 1250, accuracy: 91.2 },
  { agent: 'Workflow', requests: 189, avgTime: 180, accuracy: 97.8 },
];

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

function SimpleBarChart({ data }: { data: typeof mockDailyStats }) {
  const maxRequests = Math.max(...data.map(d => d.requests));
  
  return (
    <div className="flex items-end justify-between h-40 gap-2">
      {data.map((day, index) => (
        <div key={index} className="flex-1 flex flex-col items-center">
          <div className="w-full flex flex-col items-center">
            <div 
              className="w-full bg-primary-500 rounded-t"
              style={{ height: `${(day.requests / maxRequests) * 120}px` }}
            />
            <div 
              className="w-full bg-urgent-500 rounded-t -mt-1"
              style={{ height: `${(day.escalations / maxRequests) * 120}px` }}
            />
          </div>
          <span className="text-xs text-gray-500 mt-2">{day.date}</span>
        </div>
      ))}
    </div>
  );
}

export default function AnalyticsPage() {
  const { stats, loading } = useEscalationStats(false);

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation pendingReviews={stats?.pending_reviews || 0} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Monitor system performance, usage metrics, and AI evaluation results.
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <MetricCard 
            title="Total Requests (Today)" 
            value={stats?.total_requests || 234}
            subtitle="↑ 12% from yesterday"
            variant="default"
          />
          <MetricCard 
            title="Avg Response Time" 
            value={`${stats?.average_response_time_ms || 345}ms`}
            subtitle="Target: <500ms"
            variant="success"
          />
          <MetricCard 
            title="Escalation Rate" 
            value={`${((stats?.escalation_rate || 0.08) * 100).toFixed(1)}%`}
            subtitle="Target: <10%"
            variant={stats?.escalation_rate && stats.escalation_rate > 0.1 ? 'warning' : 'success'}
          />
          <MetricCard 
            title="Critical Findings" 
            value={stats?.critical_findings_today || 3}
            subtitle="Requires immediate attention"
            variant={stats?.critical_findings_today && stats.critical_findings_today > 0 ? 'error' : 'success'}
          />
        </div>

        {/* Charts Row */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Daily Activity Chart */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Daily Activity</h2>
                <div className="flex items-center gap-4 text-xs">
                  <span className="flex items-center">
                    <span className="w-3 h-3 bg-primary-500 rounded mr-1"></span>
                    Requests
                  </span>
                  <span className="flex items-center">
                    <span className="w-3 h-3 bg-urgent-500 rounded mr-1"></span>
                    Escalations
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <SimpleBarChart data={mockDailyStats} />
            </CardContent>
          </Card>

          {/* Agent Performance */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900">Agent Performance</h2>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockAgentMetrics.map((agent, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">{agent.agent} Agent</p>
                      <p className="text-xs text-gray-500">{agent.requests} requests today</p>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="text-sm font-medium">{agent.avgTime}ms</p>
                        <p className="text-xs text-gray-500">avg time</p>
                      </div>
                      <Badge variant={agent.accuracy >= 95 ? 'success' : agent.accuracy >= 90 ? 'info' : 'urgent'}>
                        {agent.accuracy}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Model Evaluation Section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Model Evaluation Metrics</h2>
              <Badge variant="info">Last updated: 1h ago</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              {/* MedGemma */}
              <div className="p-4 border border-gray-200 rounded-lg">
                <h3 className="font-medium text-gray-900 mb-3">MedGemma 4B</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Medical QA Accuracy</span>
                    <span className="font-medium">87.3%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Triage Accuracy</span>
                    <span className="font-medium">92.1%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Avg Latency</span>
                    <span className="font-medium">234ms</span>
                  </div>
                </div>
              </div>

              {/* CXR Foundation */}
              <div className="p-4 border border-gray-200 rounded-lg">
                <h3 className="font-medium text-gray-900 mb-3">CXR Foundation</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">AUC-ROC</span>
                    <span className="font-medium">0.94</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Sensitivity</span>
                    <span className="font-medium">89.5%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Specificity</span>
                    <span className="font-medium">91.2%</span>
                  </div>
                </div>
              </div>

              {/* Path Foundation */}
              <div className="p-4 border border-gray-200 rounded-lg">
                <h3 className="font-medium text-gray-900 mb-3">Path Foundation</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Classification Acc</span>
                    <span className="font-medium">85.8%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">IoU Score</span>
                    <span className="font-medium">0.78</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Avg Latency</span>
                    <span className="font-medium">1.2s</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Drift Monitoring */}
            <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-start gap-3">
                <span className="text-yellow-500 text-xl">⚠️</span>
                <div>
                  <h4 className="font-medium text-yellow-800">Drift Monitoring</h4>
                  <p className="text-sm text-yellow-700 mt-1">
                    Input distribution shift detected for pathology images. 
                    KL divergence: 0.12 (threshold: 0.1). Consider retraining or reviewing recent cases.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
