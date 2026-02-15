'use client';

import React, { useEffect } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center px-6">
      <div className="max-w-md w-full text-center">
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-full">
            <AlertCircle className="w-10 h-10 text-red-400" />
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-3">Something went wrong</h2>

        <p className="text-white/60 text-sm mb-2">
          An unexpected error occurred in the application.
        </p>

        {error.digest && (
          <p className="text-white/30 text-xs font-mono mb-6">
            Error ID: {error.digest}
          </p>
        )}

        {process.env.NODE_ENV === 'development' && (
          <pre className="text-left text-xs text-red-300/80 bg-red-500/5 border border-red-500/10 rounded-lg p-4 mb-6 overflow-auto max-h-40">
            {error.message}
          </pre>
        )}

        <button
          onClick={reset}
          className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl font-medium text-white hover:shadow-lg hover:shadow-purple-500/30 transition"
        >
          <RefreshCw className="w-4 h-4" />
          Try Again
        </button>
      </div>
    </div>
  );
}
