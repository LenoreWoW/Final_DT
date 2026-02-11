'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Atom,
  Loader2,
  AlertCircle,
  Sparkles,
  ArrowRight,
  Clock,
  Zap,
} from 'lucide-react';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';
import { twinService, Twin } from '@/lib/api';

export default function DashboardIndexPage() {
  const [twins, setTwins] = useState<Twin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTwins = async () => {
      try {
        const data = await twinService.listTwins();
        setTwins(data);
      } catch (err) {
        console.error('Failed to fetch twins:', err);
        setError('Unable to load your twins. Please log in and try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchTwins();
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <QuantumParticles />
      <GlassNavigation />

      <main className="relative z-10 container mx-auto px-6 pt-24 pb-16">
        <motion.header
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            >
              <Atom className="w-8 h-8 text-cyan-400" />
            </motion.div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-white via-blue-200 to-cyan-200 bg-clip-text text-transparent">
              Your Digital Twins
            </h1>
          </div>
          <p className="text-white/60 max-w-2xl">
            Manage and explore all the quantum-powered digital twins you have created.
          </p>
        </motion.header>

        {loading && (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-10 h-10 text-blue-400 animate-spin mb-4" />
            <p className="text-white/50">Loading your twins...</p>
          </div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-start gap-3 p-6 bg-red-500/10 border border-red-500/20 rounded-xl max-w-lg mx-auto"
          >
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-300 text-sm">{error}</p>
              <Link
                href="/login"
                className="text-blue-400 hover:text-blue-300 text-sm font-medium mt-2 inline-block transition"
              >
                Go to Login
              </Link>
            </div>
          </motion.div>
        )}

        {!loading && !error && twins.length === 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center py-32 text-center"
          >
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center mb-6 border border-blue-500/30">
              <Sparkles className="w-10 h-10 text-blue-400" />
            </div>
            <h2 className="text-xl font-semibold text-white/80 mb-2">No Twins Yet</h2>
            <p className="text-white/40 mb-6 max-w-md">
              You have not created any digital twins yet. Head to the Builder to describe a system and generate your first quantum twin.
            </p>
            <Link
              href="/builder"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl font-medium text-white hover:shadow-lg hover:shadow-purple-500/30 transition"
            >
              <Sparkles className="w-4 h-4" />
              Start Building
            </Link>
          </motion.div>
        )}

        {!loading && !error && twins.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {twins.map((twin, idx) => (
              <motion.div
                key={twin.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.08 }}
              >
                <Link href={`/dashboard/${twin.id}`}>
                  <div className="group bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm hover:border-cyan-500/30 hover:shadow-lg hover:shadow-cyan-500/10 transition-all cursor-pointer">
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="text-lg font-semibold text-white group-hover:text-cyan-200 transition">
                        {twin.name}
                      </h3>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium border ${
                          twin.status === 'active'
                            ? 'bg-green-500/20 text-green-400 border-green-500/30'
                            : twin.status === 'generating'
                            ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                            : twin.status === 'draft'
                            ? 'bg-white/10 text-white/60 border-white/20'
                            : 'bg-red-500/20 text-red-400 border-red-500/30'
                        }`}
                      >
                        {twin.status?.toUpperCase()}
                      </span>
                    </div>

                    <p className="text-sm text-white/50 mb-4 line-clamp-2">
                      {twin.description || 'No description provided.'}
                    </p>

                    <div className="flex items-center justify-between text-xs text-white/30">
                      <div className="flex items-center gap-3">
                        {twin.domain && (
                          <span className="flex items-center gap-1">
                            <Zap className="w-3 h-3" />
                            {twin.domain}
                          </span>
                        )}
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {new Date(twin.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <ArrowRight className="w-4 h-4 text-white/20 group-hover:text-cyan-400 transition" />
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
