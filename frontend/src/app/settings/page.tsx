'use client';

import React, { useState } from 'react';
import { Navigation } from '@/components/Navigation';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';

interface SettingSection {
  id: string;
  title: string;
  description: string;
  settings: Setting[];
}

interface Setting {
  id: string;
  label: string;
  description: string;
  type: 'toggle' | 'select' | 'text' | 'number';
  value: string | number | boolean;
  options?: { label: string; value: string }[];
}

const defaultSections: SettingSection[] = [
  {
    id: 'general',
    title: 'General Settings',
    description: 'Configure general application behavior',
    settings: [
      {
        id: 'polling_interval',
        label: 'Polling Interval',
        description: 'How often to refresh data (in seconds)',
        type: 'select',
        value: '30',
        options: [
          { label: '15 seconds', value: '15' },
          { label: '30 seconds', value: '30' },
          { label: '60 seconds', value: '60' },
          { label: '5 minutes', value: '300' },
        ],
      },
      {
        id: 'notifications',
        label: 'Enable Notifications',
        description: 'Show desktop notifications for critical findings',
        type: 'toggle',
        value: true,
      },
      {
        id: 'language',
        label: 'Language',
        description: 'Interface language',
        type: 'select',
        value: 'en',
        options: [
          { label: 'English', value: 'en' },
          { label: 'Spanish', value: 'es' },
          { label: 'French', value: 'fr' },
        ],
      },
    ],
  },
  {
    id: 'ai',
    title: 'AI Configuration',
    description: 'Configure AI model behavior and thresholds',
    settings: [
      {
        id: 'confidence_threshold',
        label: 'Confidence Threshold',
        description: 'Minimum confidence score before escalation (%)',
        type: 'number',
        value: 85,
      },
      {
        id: 'escalation_enabled',
        label: 'Auto-Escalation',
        description: 'Automatically escalate low-confidence results',
        type: 'toggle',
        value: true,
      },
      {
        id: 'model_version',
        label: 'MedGemma Model',
        description: 'Select the MedGemma model version',
        type: 'select',
        value: 'medgemma-4b-it',
        options: [
          { label: 'MedGemma 4B IT', value: 'medgemma-4b-it' },
          { label: 'MedGemma 27B IT', value: 'medgemma-27b-it' },
        ],
      },
    ],
  },
  {
    id: 'security',
    title: 'Security & Privacy',
    description: 'Configure security and HIPAA compliance settings',
    settings: [
      {
        id: 'audit_logging',
        label: 'Audit Logging',
        description: 'Log all actions for HIPAA compliance',
        type: 'toggle',
        value: true,
      },
      {
        id: 'session_timeout',
        label: 'Session Timeout',
        description: 'Auto-logout after inactivity (minutes)',
        type: 'number',
        value: 30,
      },
      {
        id: 'phi_detection',
        label: 'PHI Detection',
        description: 'Automatically detect and protect PHI in inputs',
        type: 'toggle',
        value: true,
      },
    ],
  },
];

function SettingRow({ setting, onUpdate }: { setting: Setting; onUpdate: (id: string, value: any) => void }) {
  const renderInput = () => {
    switch (setting.type) {
      case 'toggle':
        return (
          <button
            onClick={() => onUpdate(setting.id, !setting.value)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              setting.value ? 'bg-primary-600' : 'bg-gray-300'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                setting.value ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        );
      case 'select':
        return (
          <select
            value={setting.value as string}
            onChange={(e) => onUpdate(setting.id, e.target.value)}
            className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {setting.options?.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        );
      case 'number':
        return (
          <input
            type="number"
            value={setting.value as number}
            onChange={(e) => onUpdate(setting.id, parseInt(e.target.value))}
            className="w-24 px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        );
      case 'text':
        return (
          <input
            type="text"
            value={setting.value as string}
            onChange={(e) => onUpdate(setting.id, e.target.value)}
            className="w-48 px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        );
    }
  };

  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
      <div>
        <p className="font-medium text-gray-900">{setting.label}</p>
        <p className="text-sm text-gray-500">{setting.description}</p>
      </div>
      <div className="ml-4">{renderInput()}</div>
    </div>
  );
}

export default function SettingsPage() {
  const [sections, setSections] = useState(defaultSections);
  const [saved, setSaved] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const handleUpdate = (sectionId: string, settingId: string, value: any) => {
    setSections(prev => prev.map(section => {
      if (section.id !== sectionId) return section;
      return {
        ...section,
        settings: section.settings.map(setting => {
          if (setting.id !== settingId) return setting;
          return { ...setting, value };
        }),
      };
    }));
    setHasChanges(true);
    setSaved(false);
  };

  const handleSave = () => {
    // In production, this would call the API to save settings
    console.log('Saving settings:', sections);
    setSaved(true);
    setHasChanges(false);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    setSections(defaultSections);
    setHasChanges(false);
    setSaved(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
            <p className="text-gray-600 mt-1">
              Configure application settings and preferences.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {hasChanges && (
              <Badge variant="urgent">Unsaved changes</Badge>
            )}
            <Button variant="outline" onClick={handleReset} disabled={!hasChanges}>
              Reset
            </Button>
            <Button onClick={handleSave} disabled={!hasChanges}>
              Save Changes
            </Button>
          </div>
        </div>

        {saved && (
          <Alert variant="success" className="mb-6">
            Settings saved successfully.
          </Alert>
        )}

        <div className="space-y-6">
          {sections.map((section) => (
            <Card key={section.id}>
              <CardHeader>
                <h2 className="text-lg font-semibold text-gray-900">{section.title}</h2>
                <p className="text-sm text-gray-500">{section.description}</p>
              </CardHeader>
              <CardContent>
                {section.settings.map((setting) => (
                  <SettingRow
                    key={setting.id}
                    setting={setting}
                    onUpdate={(id, value) => handleUpdate(section.id, id, value)}
                  />
                ))}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* System Info */}
        <Card className="mt-6">
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">System Information</h2>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Version</p>
                <p className="font-medium">MedAI Compass v1.0.0</p>
              </div>
              <div>
                <p className="text-gray-500">API Endpoint</p>
                <p className="font-medium font-mono text-xs">http://localhost:8000</p>
              </div>
              <div>
                <p className="text-gray-500">Models Loaded</p>
                <p className="font-medium">MedGemma 4B, CXR Foundation, Path Foundation</p>
              </div>
              <div>
                <p className="text-gray-500">Environment</p>
                <Badge variant="info">Development</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Danger Zone */}
        <Card className="mt-6 border-emergency-200">
          <CardHeader>
            <h2 className="text-lg font-semibold text-emergency-700">Danger Zone</h2>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900">Clear All Data</p>
                <p className="text-sm text-gray-500">
                  Remove all cached data and reset the application. This cannot be undone.
                </p>
              </div>
              <Button variant="danger" size="sm">
                Clear Data
              </Button>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
