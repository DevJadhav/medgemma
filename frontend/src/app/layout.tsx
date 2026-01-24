import type { Metadata } from 'next';
import { Inter, Playfair_Display } from 'next/font/google';
import './globals.css';
import clsx from 'clsx';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const playfair = Playfair_Display({ subsets: ['latin'], variable: '--font-playfair' });

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
    <html lang="en" className="h-full antialiased">
      <body className={clsx(inter.variable, playfair.variable, "h-full font-sans bg-subtle-gradient text-foreground selection:bg-primary selection:text-primary-foreground")}>{children}</body>
    </html>
  );
}
