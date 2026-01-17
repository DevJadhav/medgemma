import { PatientChat } from '@/components/chat/PatientChat';
import { Navigation } from '@/components/Navigation';

export default function ChatPage() {
  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Navigation />
      <div className="flex-1 overflow-hidden">
        <PatientChat className="h-full max-w-4xl mx-auto bg-white shadow-lg" />
      </div>
    </div>
  );
}
