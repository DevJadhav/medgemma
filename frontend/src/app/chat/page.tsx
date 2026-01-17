import { PatientChat } from '@/components/chat/PatientChat';

export default function ChatPage() {
  return (
    <div className="h-screen bg-gray-50">
      <PatientChat className="h-full max-w-4xl mx-auto bg-white shadow-lg" />
    </div>
  );
}
