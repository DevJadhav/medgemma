'use client';

import React, { useState } from 'react';
import { Navigation } from '@/components/Navigation';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useWorkflow } from '@/hooks/useApi';
import { Alert } from '@/components/ui/Alert';

type Tab = 'scheduling' | 'documentation' | 'prior_auth';

export default function WorkflowPage() {
    const [activeTab, setActiveTab] = useState<Tab>('scheduling');
    const { processWorkflow, loading, error } = useWorkflow();
    const [result, setResult] = useState<any>(null);

    // Form states
    const [patientId, setPatientId] = useState('');
    const [details, setDetails] = useState('');

    const handleSubmit = async () => {
        if (!patientId || !details) return;

        const response = await processWorkflow({
            request_type: activeTab,
            patient_id: patientId,
            data: {
                text: details,
                // Mock data for specific types
                date: new Date().toISOString(),
                provider_id: 'provider-123'
            }
        });

        if (response) {
            setResult(response);
        }
    };

    return (
        <div className="min-h-screen bg-subtle-gradient">
            <Navigation />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
                {/* Header */}
                <div>
                    <h1 className="text-3xl font-display font-semibold text-foreground tracking-tight">Clinical Workflow</h1>
                    <p className="text-muted-foreground mt-2">Automate administrative tasks with the Workflow Agent.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column: Input */}
                    <div className="lg:col-span-2 space-y-6">

                        {/* Tabs */}
                        <div className="flex p-1 bg-secondary/50 backdrop-blur-sm rounded-xl border border-border/40">
                            {(['scheduling', 'documentation', 'prior_auth'] as Tab[]).map((tab) => (
                                <button
                                    key={tab}
                                    onClick={() => { setActiveTab(tab); setResult(null); }}
                                    className={`flex-1 py-2.5 text-sm font-medium rounded-lg transition-all ${activeTab === tab
                                            ? 'bg-background text-foreground shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
                                        }`}
                                >
                                    {tab === 'prior_auth' ? 'Prior Auth' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                                </button>
                            ))}
                        </div>

                        <Card variant="glass" className="border-border/40">
                            <CardHeader>
                                <h2 className="text-xl font-display font-semibold text-foreground">
                                    {activeTab === 'scheduling' && 'Schedule Appointment'}
                                    {activeTab === 'documentation' && 'Generate Documentation'}
                                    {activeTab === 'prior_auth' && 'Request Authorization'}
                                </h2>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">Patient ID</label>
                                    <input
                                        type="text"
                                        value={patientId}
                                        onChange={(e) => setPatientId(e.target.value)}
                                        className="w-full p-2.5 bg-white/50 border border-input rounded-lg focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                                        placeholder="e.g. PT-123456"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        {activeTab === 'scheduling' && 'Appointment Details'}
                                        {activeTab === 'documentation' && 'Clinical Notes'}
                                        {activeTab === 'prior_auth' && 'Procedure/Medication Details'}
                                    </label>
                                    <textarea
                                        value={details}
                                        onChange={(e) => setDetails(e.target.value)}
                                        className="w-full h-32 p-3 bg-white/50 border border-input rounded-lg focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all resize-none"
                                        placeholder="Enter details..."
                                    />
                                </div>

                                <Button
                                    onClick={handleSubmit}
                                    loading={loading}
                                    disabled={!patientId || !details}
                                    className="w-full"
                                >
                                    Process Request
                                </Button>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Right Column: Output */}
                    <div>
                        <Card variant="glass" className="h-full border-border/40">
                            <CardHeader>
                                <h2 className="text-xl font-display font-semibold text-foreground">Agent Output</h2>
                            </CardHeader>
                            <CardContent>
                                {loading ? (
                                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground animate-pulse">
                                        <div className="w-12 h-12 rounded-full bg-primary/10 mb-4"></div>
                                        <p>Processing request...</p>
                                    </div>
                                ) : result ? (
                                    <div className="space-y-4 animate-fade-in">
                                        <Alert variant="success" className="border-emerald-200 bg-emerald-50/50">
                                            Request Completed Successfully
                                        </Alert>

                                        <div className="p-4 bg-white/50 rounded-xl border border-border/50 space-y-2 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Status:</span>
                                                <span className="font-medium text-foreground capitalize">{result.status}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Request ID:</span>
                                                <span className="font-mono text-xs text-foreground">{result.request_id.slice(0, 8)}...</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Time:</span>
                                                <span className="font-medium text-foreground">{result.processing_time_ms.toFixed(0)}ms</span>
                                            </div>
                                        </div>

                                        <div className="bg-slate-950 text-slate-50 p-4 rounded-xl text-xs font-mono overflow-auto max-h-60">
                                            <pre>{JSON.stringify(result.result || result, null, 2)}</pre>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground/50">
                                        <p>No active output</p>
                                    </div>
                                )}

                                {error && (
                                    <Alert variant="error" className="mt-4">
                                        {error}
                                    </Alert>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </main>
        </div>
    );
}
