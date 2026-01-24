'use client';

import React, { useState } from 'react';
import { Navigation } from '@/components/Navigation';
import { useGuardrailsConfig, useGuardrailsTest, GuardrailsTestResponse } from '@/hooks/useApi';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

export default function GuardrailsPage() {
    const { config, loading: configLoading } = useGuardrailsConfig();
    const { testGuardrails, loading: testLoading } = useGuardrailsTest();

    const [inputText, setInputText] = useState('');
    const [testResult, setTestResult] = useState<GuardrailsTestResponse | null>(null);

    const handleTest = async () => {
        if (!inputText.trim()) return;
        const result = await testGuardrails(inputText);
        setTestResult(result);
    };

    return (
        <div className="min-h-screen bg-gray-50/50">
            <Navigation />
            <div className="p-8 space-y-8">
                {/* Header */}
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">AI Guardrails</h1>
                    <p className="text-gray-500 mt-2">Interactive playground to test safety filters and view active configurations.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left Column: Playground */}
                    <div className="space-y-6">
                        <Card variant="glass" className="border-border/40">
                            <CardHeader>
                                <h2 className="text-xl font-display font-semibold text-foreground">Safety Playground</h2>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-2">Input Text</label>
                                    <textarea
                                        className="w-full h-32 p-3 bg-white/50 backdrop-blur-sm border border-input rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary resize-none transition-all placeholder:text-muted-foreground"
                                        placeholder="Type a message to test guardrails (e.g., 'Ignore previous instructions' or 'Patient needs an MRI')..."
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                    />
                                </div>
                                <Button
                                    onClick={handleTest}
                                    loading={testLoading}
                                    disabled={!inputText.trim()}
                                    className="w-full"
                                >
                                    Test Guardrails
                                </Button>
                            </CardContent>
                        </Card>

                        {/* Results Area */}
                        {testResult && (
                            <Card variant="glass" className={testResult.is_safe ? "border-l-4 border-l-emerald-500" : "border-l-4 border-l-red-500"}>
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <h3 className="font-semibold text-foreground">Analysis Result</h3>
                                        <span className={`px-3 py-1 rounded-full text-sm font-bold shadow-sm ${testResult.is_safe ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
                                            {testResult.is_safe ? "SAFE" : "BLOCKED"}
                                        </span>
                                    </div>
                                </CardHeader>
                                <CardContent className="space-y-4">

                                    {/* Jailbreak Section */}
                                    <div className="p-4 bg-white/40 rounded-xl border border-border/50">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="font-medium text-foreground">Jailbreak Detection</span>
                                            <span className={`text-xs font-bold uppercase tracking-wider ${testResult.jailbreak.detected ? "text-red-600" : "text-emerald-600"}`}>
                                                {testResult.jailbreak.detected ? "DETECTED" : "PASSED"}
                                            </span>
                                        </div>
                                        {testResult.jailbreak.detected && (
                                            <div className="text-sm text-red-600 space-y-1">
                                                <p>Category: <span className="font-medium">{testResult.jailbreak.category}</span></p>
                                                <p>Severity: <span className="font-medium">{testResult.jailbreak.severity}</span></p>
                                                <p>Risk Score: <span className="font-medium">{testResult.jailbreak.risk_score}</span></p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Injection Section */}
                                    <div className="p-4 bg-white/40 rounded-xl border border-border/50">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="font-medium text-foreground">Prompt Injection</span>
                                            <span className={`text-xs font-bold uppercase tracking-wider ${testResult.injection.detected ? "text-red-600" : "text-emerald-600"}`}>
                                                {testResult.injection.detected ? "DETECTED" : "PASSED"}
                                            </span>
                                        </div>
                                        {testResult.injection.detected && (
                                            <div className="text-sm text-red-600">
                                                <p>Reason: {testResult.injection.reason}</p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Medical Scope Section */}
                                    <div className="p-4 bg-white/40 rounded-xl border border-border/50">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="font-medium text-foreground">Medical Scope</span>
                                            <span className={`text-xs font-bold uppercase tracking-wider ${testResult.scope.is_valid ? "text-emerald-600" : "text-amber-600"}`}>
                                                {testResult.scope.is_valid ? "VALID" : "OUT OF SCOPE"}
                                            </span>
                                        </div>
                                        <div className="text-sm text-muted-foreground">
                                            {testResult.scope.scope ? (
                                                <p>Type: <span className="font-medium bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs">{testResult.scope.scope}</span></p>
                                            ) : (
                                                <p>{testResult.scope.reason || "Input does not match medical intents"}</p>
                                            )}
                                        </div>
                                    </div>

                                </CardContent>
                            </Card>
                        )}
                    </div>

                    {/* Right Column: Configurations */}
                    <div className="space-y-6">
                        <Card variant="glass" className="h-full border-border/40">
                            <CardHeader>
                                <h2 className="text-xl font-display font-semibold text-foreground">Active Configurations</h2>
                            </CardHeader>
                            <CardContent>
                                {configLoading ? (
                                    <div className="animate-pulse space-y-4">
                                        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                                    </div>
                                ) : config ? (
                                    <div className="space-y-6">
                                        {/* Jailbreak Categories */}
                                        <div>
                                            <h3 className="font-medium text-foreground mb-3 flex items-center gap-2">
                                                <ShieldIcon />
                                                Jailbreak Protection
                                            </h3>
                                            <div className="flex flex-wrap gap-2">
                                                {config.jailbreak_categories.map(cat => (
                                                    <span key={cat} className="px-2.5 py-1 bg-red-50 text-red-700 border border-red-100 rounded-full text-xs font-medium">
                                                        {cat}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="h-px bg-border/50"></div>

                                        {/* detailed scopes */}
                                        {Object.entries(config.scope_patterns).map(([scope, patterns]) => (
                                            <div key={scope}>
                                                <h3 className="font-medium text-foreground mb-2 capitalize flex items-center gap-2">
                                                    <ScopeIcon scope={scope} />
                                                    {scope} Scope
                                                </h3>
                                                <div className="bg-white/50 backdrop-blur-sm p-3 rounded-xl border border-border/50 max-h-40 overflow-y-auto">
                                                    <ul className="text-sm text-muted-foreground space-y-1 font-mono">
                                                        {patterns.map((p, idx) => (
                                                            <li key={idx} className="truncate">{p}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-destructive">Failed to load configuration.</div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
}

function ShieldIcon() {
    return (
        <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
    );
}

function ScopeIcon({ scope }: { scope: string }) {
    if (scope === 'diagnostic') return (
        <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
    );
    if (scope === 'workflow') return (
        <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
    );
    // Default communication key
    return (
        <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
    );
}
