'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Atom, Zap, Target, BarChart3, Sparkles, ArrowRight, ChevronDown } from 'lucide-react';
import Link from 'next/link';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';
import ScrollReveal, { StaggerReveal, TextReveal } from '@/components/animations/ScrollReveal';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* WebGL Particle Background */}
      <QuantumParticles />

      {/* Glass Navigation */}
      <GlassNavigation />

      {/* Hero Section - Full Viewport */}
      <section className="min-h-screen flex flex-col items-center justify-center relative">
        <div className="container mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-5xl mx-auto"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/20 border border-blue-500/30 mb-8"
          >
            <Atom className="w-5 h-5 text-blue-400 animate-spin" style={{ animationDuration: '4s' }} />
            <span className="text-blue-300 font-medium">Quantum-Powered Digital Twins</span>
          </motion.div>

          <motion.h1
            className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
          >
            <span className="bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent">
              Build a <em className="not-italic text-cyan-300">Second</em> World
            </span>
          </motion.h1>

          <motion.p
            className="text-2xl md:text-3xl text-white/70 mb-4 leading-relaxed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Describe any reality. Simulate infinite futures.
          </motion.p>

          <motion.p
            className="text-lg text-white/60 max-w-3xl mx-auto mb-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
          >
            Describe any system—a human body, a city, a battlefield, an ecosystem—and receive a fully functional
            quantum-powered digital twin. Simulate thousands of scenarios, predict futures, optimize strategies,
            and discover hidden patterns.
          </motion.p>

          {/* Key Stats */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-2">
                1000x
              </div>
              <div className="text-white/70">Faster Optimization</div>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
                +13%
              </div>
              <div className="text-white/70">Accuracy Improvement</div>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                ∞
              </div>
              <div className="text-white/70">Domains Supported</div>
            </div>
          </div>
        </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-10 left-1/2 -translate-x-1/2"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.5 }}
        >
          <div className="flex flex-col items-center gap-2 text-white/50">
            <span className="text-sm">Scroll to explore</span>
            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            >
              <ChevronDown className="w-6 h-6" />
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Two Pillars Section */}
      <section className="py-20">
        <div className="container mx-auto px-6">
          <ScrollReveal direction="fade">
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-4">
              Two Ways to Experience <span className="text-cyan-400">Quantum Power</span>
            </h2>
            <p className="text-white/60 text-center max-w-2xl mx-auto mb-16">
              Build your own quantum twins or explore proven quantum advantage in healthcare
            </p>
          </ScrollReveal>

          <StaggerReveal className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {/* Pillar 1: Universal Builder */}
            <Link href="/builder">
              <div className="group h-full bg-gradient-to-br from-blue-600/20 to-cyan-600/20 border border-blue-500/30 rounded-2xl p-8 hover:border-blue-400 hover:shadow-2xl hover:shadow-blue-500/50 transition-all cursor-pointer transform hover:scale-105 hover:-translate-y-2 duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                    <Sparkles className="w-7 h-7 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold">Universal Builder</h2>
                </div>

                <p className="text-white/70 text-lg mb-6 leading-relaxed">
                  The main application where users build quantum digital twins for ANY domain.
                  No pre-built knowledge needed. Just describe your system and watch it come to life.
                </p>

                <div className="space-y-3 mb-6">
                  <div className="flex items-start gap-2">
                    <Target className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">Works for business, personal, scientific, hypothetical scenarios</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">Quantum algorithms optimize, simulate, and predict automatically</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <BarChart3 className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">Interactive dashboard with scenario branching</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-blue-400 group-hover:gap-4 transition-all font-medium">
                  Start Building
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </Link>

            {/* Pillar 2: Quantum Showcase */}
            <Link href="/showcase">
              <div className="group h-full bg-gradient-to-br from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-2xl p-8 hover:border-purple-400 hover:shadow-2xl hover:shadow-purple-500/50 transition-all cursor-pointer transform hover:scale-105 hover:-translate-y-2 duration-300">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <Atom className="w-7 h-7 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold">Quantum Showcase</h2>
                </div>

                <p className="text-white/70 text-lg mb-6 leading-relaxed">
                  Proof that quantum beats classical. Healthcare case study with side-by-side comparisons,
                  benchmark data, and interactive demos.
                </p>

                <div className="space-y-3 mb-6">
                  <div className="flex items-start gap-2">
                    <Target className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">6 healthcare modules with proven quantum advantage</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">Fair comparisons with optimized classical baselines</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <BarChart3 className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
                    <span className="text-white/70">Statistical validation with p &lt; 0.001</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-purple-400 group-hover:gap-4 transition-all font-medium">
                  See the Proof
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </Link>
          </StaggerReveal>
        </div>
      </section>

      {/* Footer Info */}
      <footer className="py-20">
        <ScrollReveal direction="fade">
          <div className="text-center text-white/50 max-w-4xl mx-auto px-6">
            <p className="text-sm mb-4">
              Powered by Qiskit, PennyLane, Qiskit Aer Simulator • Thesis Defense Ready • Reproducible Results
            </p>
            <div className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent mb-4" />
            <p className="text-xs text-white/30">
              Immersive. Intelligent. Infinite possibilities.
            </p>
          </div>
        </ScrollReveal>
      </footer>
    </div>
  );
}

