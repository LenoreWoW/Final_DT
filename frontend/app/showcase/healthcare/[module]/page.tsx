'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  Play,
  Loader2,
  Zap,
  Clock,
  Target,
  ArrowRight,
  BarChart3,
  Code2,
  TrendingUp,
  ChevronLeft,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import Link from 'next/link';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';
import QASMViewer from '@/components/quantum/QASMViewer';
import { benchmarkService } from '@/lib/api';

const MODULE_INFO: Record<
  string,
  {
    title: string;
    description: string;
    problem: string;
    classical: string;
    quantum: string;
    icon: string;
  }
> = {
  personalized_medicine: {
    title: 'Personalized Medicine',
    description: 'Optimize cancer treatment plans using quantum algorithms to explore exponentially more drug combinations.',
    problem: 'Finding the optimal treatment combination from millions of possibilities for individual patients based on their genomic profile, biomarkers, and tumor characteristics.',
    classical: 'Genetic Algorithm with Grid Search. Tests ~1,000 combinations sequentially. Limited by combinatorial explosion as variables increase.',
    quantum: 'QAOA (Quantum Approximate Optimization Algorithm). Explores 1,000,000+ combinations simultaneously via quantum superposition and interference.',
    icon: '💊',
  },
  drug_discovery: {
    title: 'Drug Discovery',
    description: 'Simulate molecular interactions at quantum scale for accurate binding affinity predictions.',
    problem: 'Screening thousands of candidate molecules against a target protein to find viable drug candidates with high binding affinity and low toxicity.',
    classical: 'Classical Molecular Dynamics simulation. Accurate but extremely slow — one molecule can take hours.',
    quantum: 'VQE (Variational Quantum Eigensolver) for molecular ground state simulation. Quantum speedup enables screening of 10,000+ molecules.',
    icon: '🧬',
  },
  medical_imaging: {
    title: 'Medical Imaging',
    description: 'Quantum-enhanced neural networks for superior tumor detection in medical scans.',
    problem: 'Detecting small tumors in CT/MRI scans with high sensitivity while minimizing false positives that lead to unnecessary biopsies.',
    classical: 'CNN (ResNet-50) trained on labeled medical images. Good accuracy but struggles with subtle features.',
    quantum: 'Quantum Neural Network with Quantum Sensing. Detects features in quantum feature space inaccessible to classical networks.',
    icon: '🏥',
  },
  genomic_analysis: {
    title: 'Genomic Analysis',
    description: 'Tensor network methods for analyzing complex multi-gene interactions in cancer.',
    problem: 'Analyzing 300+ genetic mutations to find actionable drug targets and understand multi-gene pathway interactions.',
    classical: 'PCA + Random Forest. Limited to pairwise gene interactions. Misses higher-order pathway effects.',
    quantum: 'Tree-Tensor-Networks for multi-gene pathway modeling. Captures 1000+ gene interactions simultaneously.',
    icon: '🧪',
  },
  epidemic_modeling: {
    title: 'Epidemic Modeling',
    description: 'Quantum Monte Carlo simulation for city-scale disease spread prediction.',
    problem: 'Simulating disease spread across a population of 1M+ agents with realistic contact patterns and intervention strategies.',
    classical: 'Agent-Based Modeling. Each scenario takes hours; testing 50 interventions takes days.',
    quantum: 'Quantum Monte Carlo with quantum walk simulation. 720x speedup enables real-time scenario testing during outbreaks.',
    icon: '🦠',
  },
  hospital_operations: {
    title: 'Hospital Operations',
    description: 'QAOA-optimized patient routing across multi-hospital networks.',
    problem: 'Optimally assigning 500 patients across 5 hospitals considering specialties, bed availability, transfer times, and ICU capacity.',
    classical: 'Linear Programming + Heuristics. Finds decent solutions but misses global optimum in large networks.',
    quantum: 'QAOA for combinatorial optimization of the assignment problem. Finds near-optimal solutions 100x faster.',
    icon: '🏨',
  },
};

export default function HealthcareModulePage() {
  const params = useParams();
  const rawModuleId = params.module as string;
  // URL uses kebab-case (personalized-medicine) but MODULE_INFO and API use snake_case (personalized_medicine)
  const moduleId = rawModuleId.replace(/-/g, '_');

  const info = MODULE_INFO[moduleId];
  const [precomputedResults, setPrecomputedResults] = useState<Record<string, unknown> | null>(null);
  const [liveResults, setLiveResults] = useState<Record<string, unknown> | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [qasmCircuit, setQasmCircuit] = useState<string | null>(null);
  const [showQASM, setShowQASM] = useState(false);
  const [problemSize, setProblemSize] = useState(100);

  // Fetch pre-computed results on mount
  useEffect(() => {
    benchmarkService.getModuleResults(moduleId).then((data) => {
      setPrecomputedResults(data as unknown as Record<string, unknown>);
    }).catch(console.error);
  }, [moduleId]);

  const handleRunBenchmark = useCallback(async () => {
    setIsRunning(true);
    setLiveResults(null);
    try {
      const result = await benchmarkService.runBenchmark(moduleId, {
        run_classical: true,
        run_quantum: true,
        parameters: { problem_size: problemSize },
      });
      const data = result as unknown as Record<string, unknown>;
      setLiveResults(data);

      // Extract QASM from quantum results
      const quantum = data.quantum as Record<string, unknown> | undefined;
      if (quantum?.qasm_circuit) {
        setQasmCircuit(quantum.qasm_circuit as string);
      }
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setIsRunning(false);
    }
  }, [moduleId, problemSize]);

  if (!info) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center text-white">
        <p>Module not found: {moduleId}</p>
      </div>
    );
  }

  const pc = precomputedResults as Record<string, unknown> | null;
  const comparisonData = pc
    ? [
        {
          metric: 'Time (s)',
          Classical: pc.classical_time_seconds as number,
          Quantum: pc.quantum_time_seconds as number,
        },
        {
          metric: 'Accuracy',
          Classical: ((pc.classical_accuracy as number) || 0) * 100,
          Quantum: ((pc.quantum_accuracy as number) || 0) * 100,
        },
      ]
    : [];

  const liveComparisonData =
    liveResults && (liveResults as Record<string, unknown>).classical && (liveResults as Record<string, unknown>).quantum
      ? [
          {
            metric: 'Time (s)',
            Classical: ((liveResults as Record<string, unknown>).classical as Record<string, unknown>)?.execution_time as number ?? 0,
            Quantum: ((liveResults as Record<string, unknown>).quantum as Record<string, unknown>)?.execution_time as number ?? 0,
          },
          {
            metric: 'Accuracy (%)',
            Classical: (((liveResults as Record<string, unknown>).classical as Record<string, unknown>)?.accuracy as number ?? 0) * 100,
            Quantum: (((liveResults as Record<string, unknown>).quantum as Record<string, unknown>)?.accuracy as number ?? 0) * 100,
          },
        ]
      : [];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <QuantumParticles />
      <GlassNavigation />

      <main className="relative z-10 container mx-auto px-6 pt-24 pb-16">
        {/* Back link */}
        <Link
          href="/showcase/healthcare"
          className="inline-flex items-center gap-1 text-sm text-white/50 hover:text-white/80 transition mb-6"
        >
          <ChevronLeft className="w-4 h-4" />
          Back to Healthcare Modules
        </Link>

        {/* Header */}
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-4 mb-3">
            <span className="text-4xl">{info.icon}</span>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-white via-cyan-200 to-blue-200 bg-clip-text text-transparent">
              {info.title}
            </h1>
          </div>
          <p className="text-lg text-white/60 max-w-3xl">{info.description}</p>
        </motion.div>

        {/* Problem / Classical / Quantum description */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          <motion.div
            className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Target className="w-5 h-5 text-orange-400" />
              <h3 className="font-semibold text-white">The Problem</h3>
            </div>
            <p className="text-sm text-white/60 leading-relaxed">{info.problem}</p>
          </motion.div>

          <motion.div
            className="bg-white/5 border border-red-500/20 rounded-xl p-6 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-5 h-5 text-red-400" />
              <h3 className="font-semibold text-red-300">Classical Approach</h3>
            </div>
            <p className="text-sm text-white/60 leading-relaxed">{info.classical}</p>
          </motion.div>

          <motion.div
            className="bg-white/5 border border-cyan-500/20 rounded-xl p-6 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-5 h-5 text-cyan-400" />
              <h3 className="font-semibold text-cyan-300">Quantum Approach</h3>
            </div>
            <p className="text-sm text-white/60 leading-relaxed">{info.quantum}</p>
          </motion.div>
        </div>

        {/* Pre-computed Benchmark Results */}
        {comparisonData.length > 0 && (
          <motion.div
            className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
          >
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              Benchmark Results
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <StatCard
                label="Speedup"
                value={`${pc?.speedup}x`}
                color="text-green-400"
              />
              <StatCard
                label="Accuracy Improvement"
                value={`+${((pc?.improvement as number) * 100).toFixed(0)}%`}
                color="text-cyan-400"
              />
              <StatCard
                label="Quantum Time"
                value={`${pc?.quantum_time_seconds}s`}
                color="text-blue-400"
              />
              <StatCard
                label="Classical Time"
                value={`${pc?.classical_time_seconds}s`}
                color="text-orange-400"
              />
            </div>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} barGap={8}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="metric" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 12 }} />
                  <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a0a0a',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.7)', fontSize: 12 }} />
                  <Bar dataKey="Classical" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Quantum" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}

        {/* Live Benchmark Runner */}
        <motion.div
          className="bg-gradient-to-br from-[#0a0a0a] to-[#1a1a2e] border border-cyan-500/20 rounded-xl p-6 backdrop-blur-sm mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
            <Play className="w-5 h-5 text-green-400" />
            Run Live Benchmark
          </h3>

          <div className="flex flex-col md:flex-row gap-6 items-end mb-6">
            <div className="flex-1">
              <label className="text-sm text-white/60 mb-2 block">Problem Size</label>
              <input
                type="range"
                min={10}
                max={1000}
                step={10}
                value={problemSize}
                onChange={(e) => setProblemSize(Number(e.target.value))}
                className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-cyan-500"
              />
              <div className="flex justify-between text-xs text-white/30 mt-1">
                <span>10</span>
                <span className="text-cyan-400 font-mono">{problemSize}</span>
                <span>1000</span>
              </div>
            </div>
            <button
              onClick={handleRunBenchmark}
              disabled={isRunning}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-cyan-500 rounded-xl font-medium hover:shadow-lg hover:shadow-cyan-500/30 transition disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Classical & Quantum
                </>
              )}
            </button>
          </div>

          {/* Live Results */}
          {liveComparisonData.length > 0 && (
            <div className="space-y-6">
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={liveComparisonData} barGap={8}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="metric" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 12 }} />
                    <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0a0a0a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.7)', fontSize: 12 }} />
                    <Bar dataKey="Classical" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Quantum" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {!!liveResults && !!(liveResults as Record<string, unknown>).comparison && (
                <div className="grid grid-cols-3 gap-4">
                  <StatCard
                    label="Live Speedup"
                    value={`${(((liveResults as Record<string, unknown>).comparison as Record<string, unknown>)?.speedup as number)?.toFixed(1)}x`}
                    color="text-green-400"
                  />
                  <StatCard
                    label="Accuracy Delta"
                    value={`+${((((liveResults as Record<string, unknown>).comparison as Record<string, unknown>)?.accuracy_improvement as number) * 100)?.toFixed(1)}%`}
                    color="text-cyan-400"
                  />
                  <StatCard
                    label="Quantum Won"
                    value={((liveResults as Record<string, unknown>).comparison as Record<string, unknown>)?.quantum_advantage_demonstrated ? 'Yes' : 'No'}
                    color={((liveResults as Record<string, unknown>).comparison as Record<string, unknown>)?.quantum_advantage_demonstrated ? 'text-green-400' : 'text-red-400'}
                  />
                </div>
              )}
            </div>
          )}
        </motion.div>

        {/* QASM Circuit Display */}
        {qasmCircuit && (
          <motion.div
            className="mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
          >
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <Code2 className="w-5 h-5 text-cyan-400" />
              OpenQASM Circuit
            </h3>
            <QASMViewer
              circuits={{ [moduleId]: qasmCircuit }}
              twinName={moduleId}
            />
          </motion.div>
        )}

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-white/10">
          <Link
            href="/showcase/healthcare"
            className="text-sm text-white/50 hover:text-white transition"
          >
            <ChevronLeft className="w-4 h-4 inline mr-1" />
            All Modules
          </Link>
          <Link
            href="/showcase"
            className="flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 transition"
          >
            Quantum Showcase
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </main>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-lg p-4 text-center">
      <p className="text-xs text-white/40 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}
