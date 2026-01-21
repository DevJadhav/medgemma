'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
  badge?: number;
}

interface NavigationProps {
  pendingReviews?: number;
}

// =============================================================================
// Icons (inline SVG for simplicity)
// =============================================================================

const HomeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
  </svg>
);

const ChatIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
);

const ClipboardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
  </svg>
);

const ChartIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const ImageIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
);

const SettingsIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

// =============================================================================
// Navigation Component
// =============================================================================

export function Navigation({ pendingReviews = 0 }: NavigationProps) {
  const pathname = usePathname();

  const navItems: NavItem[] = [
    { href: '/', label: 'Dashboard', icon: <HomeIcon /> },
    { href: '/chat', label: 'Patient Chat', icon: <ChatIcon /> },
    { href: '/clinician', label: 'Review Queue', icon: <ClipboardIcon />, badge: pendingReviews },
    { href: '/diagnostic', label: 'Imaging', icon: <ImageIcon /> },
    { href: '/analytics', label: 'Analytics', icon: <ChartIcon /> },
    { href: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-[0_1px_3px_rgba(0,0,0,0.05)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25 group-hover:shadow-primary-500/40 transition-shadow">
                <span className="text-xl">🏥</span>
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-lg text-gray-900 tracking-tight">
                  MedAI Compass
                </span>
                <span className="text-[10px] font-medium text-primary-600 uppercase tracking-wider">
                  Medical AI Platform
                </span>
              </div>
            </Link>
          </div>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href || 
                (item.href !== '/' && pathname.startsWith(item.href));
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                    transition-all duration-200
                    ${isActive 
                      ? 'bg-primary-50 text-primary-700 shadow-sm' 
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <span className={isActive ? 'text-primary-600' : 'text-gray-400'}>{item.icon}</span>
                  {item.label}
                  {item.badge && item.badge > 0 && (
                    <span className="ml-1 inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-bold text-white bg-gradient-to-r from-orange-500 to-red-500 rounded-full shadow-sm">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <MobileMenu navItems={navItems} currentPath={pathname} />
          </div>
        </div>
      </div>
    </nav>
  );
}

// =============================================================================
// Mobile Menu Component
// =============================================================================

interface MobileMenuProps {
  navItems: NavItem[];
  currentPath: string;
}

function MobileMenu({ navItems, currentPath }: MobileMenuProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2.5 rounded-xl text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {isOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-10 bg-black/20 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Menu */}
          <div className="absolute right-0 mt-3 w-56 bg-white rounded-2xl shadow-xl ring-1 ring-gray-100 z-20 overflow-hidden animate-fade-in">
            <div className="py-2">
              {navItems.map((item) => {
                const isActive = currentPath === item.href || 
                  (item.href !== '/' && currentPath.startsWith(item.href));
                
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setIsOpen(false)}
                    className={`
                      flex items-center gap-3 px-4 py-3 text-sm font-medium
                      transition-colors duration-150
                      ${isActive 
                        ? 'bg-primary-50 text-primary-700' 
                        : 'text-gray-700 hover:bg-gray-50'
                      }
                    `}
                  >
                    <span className={isActive ? 'text-primary-600' : 'text-gray-400'}>{item.icon}</span>
                    {item.label}
                    {item.badge && item.badge > 0 && (
                      <span className="ml-auto inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-bold text-white bg-gradient-to-r from-orange-500 to-red-500 rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// =============================================================================
// Sidebar Navigation (Alternative Layout)
// =============================================================================

export function Sidebar({ pendingReviews = 0 }: NavigationProps) {
  const pathname = usePathname();

  const navItems: NavItem[] = [
    { href: '/', label: 'Dashboard', icon: <HomeIcon /> },
    { href: '/chat', label: 'Patient Chat', icon: <ChatIcon /> },
    { href: '/clinician', label: 'Review Queue', icon: <ClipboardIcon />, badge: pendingReviews },
    { href: '/diagnostic', label: 'Imaging', icon: <ImageIcon /> },
    { href: '/analytics', label: 'Analytics', icon: <ChartIcon /> },
    { href: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  return (
    <aside className="w-64 bg-white/95 backdrop-blur-md border-r border-gray-100 min-h-screen flex flex-col">
      {/* Logo */}
      <div className="p-5 border-b border-gray-100">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
            <span className="text-xl">🏥</span>
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-gray-900">MedAI Compass</span>
            <span className="text-[10px] font-medium text-primary-600 uppercase tracking-wider">Medical AI</span>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-1 flex-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href || 
            (item.href !== '/' && pathname.startsWith(item.href));
          
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium
                transition-all duration-200
                ${isActive 
                  ? 'bg-primary-50 text-primary-700 shadow-sm' 
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }
              `}
            >
              <span className={isActive ? 'text-primary-600' : 'text-gray-400'}>{item.icon}</span>
              <span className="flex-1">{item.label}</span>
              {item.badge && item.badge > 0 && (
                <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-bold text-white bg-gradient-to-r from-orange-500 to-red-500 rounded-full">
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Status indicator */}
      <div className="p-4 border-t border-gray-100">
        <div className="bg-gray-50 rounded-xl p-4 text-xs">
          <div className="flex items-center gap-2 text-gray-700 font-medium mb-1">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            API Connected
          </div>
          <div className="text-gray-400">
            Auto-refresh every 30s
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Navigation;
