import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Navigation } from '@/components/Navigation';

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navigation />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-24 sm:py-32 bg-gradient-to-b from-primary/5 via-background to-background">
          <div className="container relative z-10 px-4 md:px-6 mx-auto">
            <div className="mx-auto max-w-2xl text-center">
              <div className="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-sm font-medium text-primary mb-8 animate-fade-in backdrop-blur-sm">
                <span className="flex h-2 w-2 rounded-full bg-primary mr-2 animate-pulse"></span>
                Powered by Google MedGemma 2
              </div>
              <h1 className="text-4xl font-display font-medium tracking-tight text-foreground sm:text-6xl mb-6 bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 animate-slide-up">
                Advanced Medical AI <br />
                <span className="text-primary italic">Re-imagined.</span>
              </h1>
              <p className="mt-6 text-lg leading-8 text-muted-foreground max-w-xl mx-auto animate-slide-up" style={{ animationDelay: '0.1s' }}>
                A sophisticated multi-agent system designed for the future of healthcare.
                Experience seamless diagnostics, patient communication, and workflow automation.
              </p>
              <div className="mt-10 flex items-center justify-center gap-x-6 animate-slide-up" style={{ animationDelay: '0.2s' }}>
                <Link href="/chat">
                  <Button size="lg" className="rounded-full px-8 text-base shadow-lg hover:shadow-primary/25 transition-all">
                    Start Patient Triage
                  </Button>
                </Link>
                <Link href="/clinician">
                  <Button variant="ghost" size="lg" className="text-base group">
                    Clinician Access <span className="inline-block transition-transform group-hover:translate-x-1">→</span>
                  </Button>
                </Link>
              </div>
            </div>
          </div>

          {/* Decorative background elements */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none opacity-30 blur-3xl -z-10">
            <div className="h-[500px] w-[500px] rounded-full bg-gradient-to-b from-primary/20 to-transparent"></div>
          </div>
        </section>

        {/* Feature Grid */}
        <section className="py-24 sm:py-32 bg-muted/50">
          <div className="container px-4 md:px-6 mx-auto">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
              <Card variant="glass" hover className="border-white/40 bg-white/60">
                <CardHeader>
                  <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold">Diagnostic Agent</h3>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">
                    Analyzing medical imaging with precision using computer vision
                    and pathology foundation models to assist in rapid diagnostics.
                  </p>
                </CardContent>
              </Card>

              <Card variant="glass" hover className="border-white/40 bg-white/60">
                <CardHeader>
                  <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-teal-50 text-teal-600">
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold">Communication Agent</h3>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">
                    Empathetic, multilingual patient triage ensuring clarity and comfort.
                    Intelligent escalation protocols for urgent care scenarios.
                  </p>
                </CardContent>
              </Card>

              <Card variant="glass" hover className="border-white/40 bg-white/60">
                <CardHeader>
                  <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold">Workflow Agent</h3>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">
                    Streamlining administrative burdens with automated clinical documentation,
                    scheduling, and prior authorization management.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Compliance Section */}
        <section className="py-24 sm:py-32">
          <div className="container px-4 md:px-6 mx-auto">
            <div className="rounded-3xl bg-primary text-primary-foreground overflow-hidden shadow-2xl relative">
              <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150"></div>
              <div className="grid lg:grid-cols-2 gap-12 p-8 md:p-16 relative z-10 items-center">
                <div>
                  <h2 className="text-3xl font-display md:text-4xl font-bold mb-6">
                    Uncompromised Safety & Compliance
                  </h2>
                  <p className="text-primary-foreground/90 text-lg mb-8 max-w-md">
                    Built with a rigorous "Safe-by-Design" philosophy. We prioritize patient data privacy and clinical accuracy above all else.
                  </p>
                  <ul className="space-y-4">
                    <li className="flex items-center gap-3">
                      <div className="bg-white/20 p-2 rounded-full"><svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg></div>
                      <span className="font-medium">HIPAA Compliant Infrastructure</span>
                    </li>
                    <li className="flex items-center gap-3">
                      <div className="bg-white/20 p-2 rounded-full"><svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg></div>
                      <span className="font-medium">Human-in-the-Loop Review</span>
                    </li>
                    <li className="flex items-center gap-3">
                      <div className="bg-white/20 p-2 rounded-full"><svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg></div>
                      <span className="font-medium">Real-time Jailbreak Detection</span>
                    </li>
                  </ul>
                </div>
                <div className="bg-white/10 rounded-2xl p-8 backdrop-blur-md border border-white/20">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="h-12 w-12 rounded-full bg-emerald-400 flex items-center justify-center text-white font-bold text-xl">99%</div>
                    <div>
                      <div className="font-bold text-xl">Safety Score</div>
                      <div className="text-primary-foreground/80 text-sm">Based on internal benchmarks</div>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="h-2 bg-black/20 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-400 w-[99%]"></div>
                    </div>
                    <div className="flex justify-between text-sm opacity-80">
                      <span>Diagnostic Accuracy</span>
                      <span>High Confidence</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t py-12 bg-gray-50/50">
        <div className="container px-4 md:px-6 mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center text-white font-bold">M</div>
            <span className="font-semibold text-lg text-foreground">MedAI Compass</span>
          </div>
          <p className="text-sm text-muted-foreground text-center md:text-right">
            © 2024 MedAI Compass. Built for the Kaggle MedGemma Impact Challenge. <br />
            <span className="text-destructive font-medium">Research Demo - Not for clinical use.</span>
          </p>
        </div>
      </footer>
    </div>
  );
}

