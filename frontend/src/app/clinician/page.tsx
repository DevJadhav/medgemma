'use client';

import { ClinicianDashboard } from '@/components/clinician/ClinicianDashboard';
import { Navigation } from '@/components/Navigation';
import { useEscalationStats } from '@/hooks/useApi';

export default function ClinicianPage() {
  const { stats } = useEscalationStats(true);
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation pendingReviews={stats?.pending_reviews || 0} />
      <ClinicianDashboard />
    </div>
  );
}
