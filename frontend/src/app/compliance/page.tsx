'use client';

import React from 'react';
import { Navigation } from '@/components/Navigation';
import { useComplianceStatus } from '@/hooks/useApi';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';

export default function CompliancePage() {
    const { status, loading, error } = useComplianceStatus();

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-8">
                <div className="bg-red-50 text-red-700 p-4 rounded-xl">
                    Error loading compliance status: {error}
                </div>
            </div>
        );
    }

    if (!status) return null;

    return (
        <div className="min-h-screen bg-gray-50/50">
            <Navigation />
            <div className="p-8 space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-gray-900 tracking-tight">HIPAA Compliance Center</h1>
                <p className="text-gray-500 mt-2">Real-time monitoring of technical, physical, and administrative safeguards.</p>
            </div>

            {/* Main Status */}
            <Card className="bg-gradient-to-br from-emerald-500 to-teal-600 text-white border-none shadow-xl">
                <CardContent className="p-8 flex items-center justify-between">
                    <div>
                        <div className="text-emerald-100 font-medium mb-1 uppercase tracking-wider text-sm">Overall Status</div>
                        <div className="text-4xl font-bold flex items-center gap-3">
                            <span className="w-4 h-4 rounded-full bg-white animate-pulse"></span>
                            {status.status.toUpperCase()}
                        </div>
                        <div className="mt-4 text-emerald-100 text-sm">
                            Last verified: {new Date(status.timestamp).toLocaleString()}
                        </div>
                    </div>
                    <div className="hidden md:block">
                        <svg className="w-32 h-32 text-white/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                    </div>
                </CardContent>
            </Card>

            {/* Safeguards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {Object.entries(status.safeguards).map(([key, value]) => (
                    <SafeguardCard key={key} title={formatTitle(key)} status={value.status} details={value.details} />
                ))}
            </div>

            {/* Policies Section (Placeholder) */}
            <h2 className="text-xl font-bold text-gray-900 mt-8">Active Policies</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <PolicyCard
                    title="Data Retention Policy"
                    description="PHI is retained for 6 years from the date of creation or the date when it was last in effect, whichever is later."
                />
                <PolicyCard
                    title="Breach Notification"
                    description="Procedures for notifying affected individuals, HHS, and the media in the event of a breach of unsecured PHI."
                />
            </div>
            </div>
        </div>
    );
}

function SafeguardCard({ title, status, details }: { title: string; status: string; details: string }) {
    const isHealthy = status === 'active';

    return (
        <Card hover className="h-full">
            <CardHeader>
                <div className="flex items-center justify-between mb-2">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isHealthy ? 'bg-emerald-100 text-emerald-600' : 'bg-red-100 text-red-600'}`}>
                        <SafeguardIcon title={title} />
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-bold ${isHealthy ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}`}>
                        {status.toUpperCase()}
                    </div>
                </div>
                <h3 className="font-semibold text-gray-900">{title}</h3>
            </CardHeader>
            <CardContent>
                <p className="text-sm text-gray-500 leading-relaxed">
                    {details}
                </p>
            </CardContent>
        </Card>
    );
}

function PolicyCard({ title, description }: { title: string; description: string }) {
    return (
        <Card className="border-l-4 border-l-primary-500">
            <CardContent className="p-6">
                <h3 className="font-semibold text-gray-900 mb-2">{title}</h3>
                <p className="text-sm text-gray-600">{description}</p>
            </CardContent>
        </Card>
    );
}

function formatTitle(key: string) {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

function SafeguardIcon({ title }: { title: string }) {
    if (title.includes('Encryption')) {
        return (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
        );
    }
    if (title.includes('Audit')) {
        return (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
        );
    }
    if (title.includes('Access')) {
        return (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
            </svg>
        );
    }
    return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
    );
}
