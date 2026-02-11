'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Activity,
  Microscope,
  Brain,
  Dna,
  Network,
  Hospital,
  ArrowRight,
  Atom,
} from 'lucide-react';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';

const healthcareModules = [
  {
    id: 'personalized-medicine',
    title: 'Personalized Medicine',
    icon: Activity,
    description:
      'Treatment optimization using QAOA vs genetic algorithms. Finds optimal treatment combinations 1000x faster by exploring millions of possibilities simultaneously.',
    quantum_advantage: '1000x faster',
    accuracy: '+13% accuracy',
    color: 'from-blue-500 to-cyan-500',
    border: 'border-blue-500/20',
    glow: 'shadow-blue-500/10',
  },
  {
    id: 'drug-discovery',
    title: 'Drug Discovery',
    icon: Microscope,
    description:
      'Molecular screening with VQE vs classical molecular dynamics. Screens drug candidates with quantum-level precision for binding affinity prediction.',
    quantum_advantage: '1000x faster',
    accuracy: 'Molecular precision',
    color: 'from-purple-500 to-pink-500',
    border: 'border-purple-500/20',
    glow: 'shadow-purple-500/10',
  },
  {
    id: 'medical-imaging',
    title: 'Medical Imaging',
    icon: Brain,
    description:
      'Tumor detection with Quantum Neural Networks vs classical CNNs. Achieves 87% accuracy compared to 74% for classical approaches in medical scan analysis.',
    quantum_advantage: '14x faster',
    accuracy: '+13% accuracy',
    color: 'from-green-500 to-emerald-500',
    border: 'border-green-500/20',
    glow: 'shadow-green-500/10',
  },
  {
    id: 'genomic-analysis',
    title: 'Genomic Analysis',
    icon: Dna,
    description:
      'Gene interaction analysis via tensor networks (TTN/MPS). Analyzes 10x more genes simultaneously, capturing complex multi-gene interactions.',
    quantum_advantage: '10x more genes',
    accuracy: 'Exponential scaling',
    color: 'from-orange-500 to-red-500',
    border: 'border-orange-500/20',
    glow: 'shadow-orange-500/10',
  },
  {
    id: 'epidemic-modeling',
    title: 'Epidemic Modeling',
    icon: Network,
    description:
      'Disease spread simulation vs agent-based models. Handles 1M+ agents in minutes for real-time pandemic response planning and intervention testing.',
    quantum_advantage: '720x faster',
    accuracy: '1M+ agents',
    color: 'from-indigo-500 to-purple-500',
    border: 'border-indigo-500/20',
    glow: 'shadow-indigo-500/10',
  },
  {
    id: 'hospital-operations',
    title: 'Hospital Operations',
    icon: Hospital,
    description:
      'Patient flow optimization with QAOA. Reduces patient wait times by 73% by finding globally optimal schedules across limited resources.',
    quantum_advantage: '-73% wait time',
    accuracy: 'Global optimum',
    color: 'from-teal-500 to-cyan-500',
    border: 'border-teal-500/20',
    glow: 'shadow-teal-500/10',
  },
];

export default function HealthcareHubPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <QuantumParticles />
      <GlassNavigation />

      <main className="relative z-10 container mx-auto px-6 pt-28 pb-20">
        {/* Hero */}
        <motion.div
          className="text-center max-w-4xl mx-auto mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#141414] border border-[#222222] mb-6">
            <Atom className="w-5 h-5 text-[#00d4ff]" />
            <span className="text-[#888888] font-medium text-sm">Healthcare Quantum Advantage</span>
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-white via-blue-200 to-cyan-200 bg-clip-text text-transparent leading-tight">
            Healthcare Modules
          </h1>

          <p className="text-lg text-[#888888] leading-relaxed max-w-3xl mx-auto">
            Six healthcare domains where quantum digital twins demonstrate measurable advantage
            over classical approaches. Each module includes head-to-head benchmarks, live demos,
            and statistically validated results.
          </p>
        </motion.div>

        {/* Module Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {healthcareModules.map((mod, index) => (
            <motion.div
              key={mod.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + index * 0.08, duration: 0.5 }}
            >
              <Link href={`/showcase/${mod.id}`}>
                <div
                  className={`group relative bg-[#141414]/80 backdrop-blur-sm border ${mod.border} rounded-xl p-6 hover:bg-[#141414] hover:scale-[1.03] hover:-translate-y-1 hover:shadow-2xl ${mod.glow} transition-all duration-300 cursor-pointer h-full flex flex-col`}
                >
                  {/* Icon */}
                  <div
                    className={`w-14 h-14 rounded-xl bg-gradient-to-br ${mod.color} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300 shadow-lg`}
                  >
                    <mod.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Title */}
                  <h3 className="text-xl font-bold mb-3 text-white group-hover:text-[#00d4ff] transition-colors">
                    {mod.title}
                  </h3>

                  {/* Description */}
                  <p className="text-[#888888] text-sm leading-relaxed mb-5 flex-1">
                    {mod.description}
                  </p>

                  {/* Stats */}
                  <div className="space-y-2 mb-5">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-[#888888]">Quantum Speed</span>
                      <span className="font-bold text-[#00ff88]">{mod.quantum_advantage}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-[#888888]">Improvement</span>
                      <span className="font-bold text-[#00d4ff]">{mod.accuracy}</span>
                    </div>
                  </div>

                  {/* CTA */}
                  <div className="pt-4 border-t border-[#222222]">
                    <span className="text-sm text-[#29648e] group-hover:text-[#00d4ff] transition-colors inline-flex items-center gap-2">
                      View Demo
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </span>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          className="mt-16 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <div className="bg-gradient-to-r from-[#29648e]/20 to-[#00d4ff]/10 border border-[#29648e]/30 rounded-xl p-10 max-w-3xl mx-auto">
            <h2 className="text-2xl font-bold mb-3">Want to see the full comparison?</h2>
            <p className="text-[#888888] mb-6">
              Visit the main Showcase page for methodology details, feature overviews,
              and a complete breakdown of quantum vs classical performance.
            </p>
            <Link
              href="/showcase"
              className="inline-flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-[#29648e] to-[#00d4ff] rounded-lg font-medium text-white hover:shadow-lg hover:shadow-[#00d4ff]/30 transition"
            >
              Full Showcase
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
