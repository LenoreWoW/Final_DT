'use client';

import React, { useEffect } from 'react';

/**
 * Global error boundary for root layout errors.
 * This must provide its own <html> and <body> tags since it replaces
 * the root layout when triggered.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Global application error:', error);
  }, [error]);

  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0a] text-[#e0e0e0] antialiased">
        <div className="min-h-screen flex items-center justify-center px-6">
          <div className="max-w-md w-full text-center">
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-full">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-10 h-10 text-red-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
            </div>

            <h2 className="text-2xl font-bold text-white mb-3">
              Critical Application Error
            </h2>

            <p className="text-white/60 text-sm mb-6">
              A critical error occurred. Please try refreshing the page.
            </p>

            {error.digest && (
              <p className="text-white/30 text-xs font-mono mb-6">
                Error ID: {error.digest}
              </p>
            )}

            <button
              onClick={reset}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl font-medium text-white hover:shadow-lg hover:shadow-purple-500/30 transition"
            >
              Refresh Page
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
