import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-primary-50 to-white">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-primary-600">
              MedAI Compass
            </h1>
            <nav className="flex gap-4">
              <Link href="/chat">
                <Button variant="outline">Patient Portal</Button>
              </Link>
              <Link href="/clinician">
                <Button>Clinician Dashboard</Button>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            HIPAA-Compliant Medical AI Platform
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Powered by Google&apos;s MedGemma, CXR Foundation, and Path Foundation models
            with intelligent multi-agent orchestration for safe, effective healthcare AI.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold text-gray-900">
                🏥 Diagnostic Agent
              </h3>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                AI-powered medical image analysis with chest X-ray and pathology
                support. Includes bounding box localization and confidence scoring.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold text-gray-900">
                💬 Communication Agent
              </h3>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Patient-facing triage and health education with multi-language
                support. Automatic escalation for emergencies and safety concerns.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold text-gray-900">
                📋 Workflow Agent
              </h3>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Clinical documentation, appointment scheduling, and prior
                authorization with intelligent task coordination.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Safety Features */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-16">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            Built for Safety & Compliance
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl mb-2">🔒</div>
              <h4 className="font-semibold text-gray-900">HIPAA Compliant</h4>
              <p className="text-sm text-gray-600">
                AES-256 encryption, audit logging, PHI protection
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">🛡️</div>
              <h4 className="font-semibold text-gray-900">Guardrails</h4>
              <p className="text-sm text-gray-600">
                Input/output validation, jailbreak detection
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">👨‍⚕️</div>
              <h4 className="font-semibold text-gray-900">Human-in-Loop</h4>
              <p className="text-sm text-gray-600">
                Clinician review for critical findings
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">📊</div>
              <h4 className="font-semibold text-gray-900">Uncertainty</h4>
              <p className="text-sm text-gray-600">
                Confidence scoring and escalation triggers
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            Ready to explore?
          </h3>
          <div className="flex justify-center gap-4">
            <Link href="/chat">
              <Button size="lg">Try Patient Chat</Button>
            </Link>
            <Link href="/clinician">
              <Button size="lg" variant="outline">
                View Clinician Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-center text-gray-500 text-sm">
            MedAI Compass • Kaggle MedGemma Impact Challenge 2024 •{' '}
            <span className="text-emergency-600">
              Not for clinical use - Demo purposes only
            </span>
          </p>
        </div>
      </footer>
    </div>
  );
}
