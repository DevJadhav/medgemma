import { PatientChat } from '@/components/chat/PatientChat';
import { Navigation } from '@/components/Navigation';

export default function ChatPage() {
  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <Navigation />
      <div className="flex-1 overflow-hidden p-4 sm:p-6">
        <div className="h-full max-w-5xl mx-auto">
          <PatientChat className="h-full bg-white rounded-2xl shadow-[0_10px_40px_-10px_rgba(0,0,0,0.1)] border border-gray-100 overflow-hidden" />
        </div>
      </div>
    </div>
  );
}
