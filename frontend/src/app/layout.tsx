import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'MedAI Compass',
  description: 'HIPAA-compliant multi-agent medical AI platform powered by Google MedGemma',
  keywords: ['medical AI', 'MedGemma', 'healthcare', 'diagnostics', 'HIPAA'],
  authors: [{ name: 'MedAI Compass Team' }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>{children}</body>
    </html>
  );
}
