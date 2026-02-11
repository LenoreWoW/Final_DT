'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  ArrowLeft,
  CheckCircle,
  Cpu,
  Scale,
  BarChart3,
  RefreshCw,
  Shield,
  Atom,
  BookOpen,
  FlaskConical,
  GitBranch,
} from 'lucide-react';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';

interface MethodologySection {
  title: string;
  content: string;
  items?: string[];
}

interface MethodologyData {
  title: string;
  overview: string;
  sections: MethodologySection[];
}

// Static fallback content used when the API is unavailable
const STATIC_METHODOLOGY: MethodologyData = {
  title: 'Benchmark Methodology',
  overview:
    'All quantum vs classical benchmarks in this platform follow a rigorous, reproducible methodology designed to ensure fair comparisons and statistically valid results.',
  sections: [
    {
      title: 'Quantum Simulation Environment',
      content:
        'All quantum circuits are executed on Qiskit Aer, a high-performance statevector and QASM simulator that faithfully reproduces the behaviour of NISQ-era quantum hardware. Circuits use realistic gate sets, noise models calibrated to real quantum hardware backends, and shot counts of 1024 or higher to reduce sampling variance.',
      items: [
        'Qiskit Aer statevector_simulator for noiseless baselines',
        'qasm_simulator with hardware-calibrated noise models for realistic estimates',
        'Minimum 1024 shots per circuit execution',
        'Error mitigation via measurement error matrices and zero-noise extrapolation',
      ],
    },
    {
      title: 'Fairness Measures',
      content:
        'Classical baselines use state-of-the-art, optimised implementations rather than naive strawman algorithms. Both quantum and classical approaches receive identical input data, the same objective functions, and equivalent computational budgets (wall-clock time or iteration count, whichever is more appropriate for the domain).',
      items: [
        'Classical algorithms sourced from well-maintained libraries (scikit-learn, SciPy, NetworkX)',
        'Same input data, same output format for side-by-side comparison',
        'Equivalent iteration budgets where applicable',
        'Warm-start advantages removed from quantum side (fair cold-start comparison)',
      ],
    },
    {
      title: 'Metrics Collected',
      content:
        'Each benchmark records execution time, solution quality, and resource usage. Where applicable, domain-specific metrics are added (e.g., binding affinity for drug discovery, AUC-ROC for medical imaging).',
      items: [
        'Wall-clock execution time (seconds)',
        'Solution quality / accuracy (domain-specific)',
        'Speedup factor (classical time / quantum time)',
        'Statistical significance via paired t-tests and confidence intervals',
        'p-values reported (threshold p < 0.001 for claimed advantages)',
        'Resource usage: qubits, circuit depth, classical memory',
      ],
    },
    {
      title: 'Reproducibility',
      content:
        'Every benchmark can be re-run from source. Seed values, hyperparameters, and dataset splits are fixed and documented. Results are stored as JSON artefacts with full provenance metadata.',
      items: [
        'Fixed random seeds for all stochastic components',
        'Hyperparameters documented in config files',
        'Datasets versioned and included in the repository',
        'Benchmark runner script: POST /api/benchmark/run/{module}',
        'Raw results available: GET /api/benchmark/results/{module}',
        'Minimum 10 independent runs per configuration for statistical validity',
      ],
    },
    {
      title: 'Statistical Validation',
      content:
        'Results are validated using paired t-tests, Wilcoxon signed-rank tests, and bootstrapped confidence intervals. Advantages are only claimed when p < 0.001 across multiple independent runs.',
      items: [
        'Paired t-test for normally distributed metrics',
        'Wilcoxon signed-rank test as non-parametric alternative',
        '95% and 99% confidence intervals reported',
        'Effect size measured via Cohen\'s d',
        'Bonferroni correction applied for multiple comparisons across modules',
      ],
    },
    {
      title: 'Limitations and Caveats',
      content:
        'Simulated quantum execution does not capture all real-hardware effects (cross-talk, T1/T2 decoherence beyond the noise model). Reported speedups include only algorithm time, not queue wait or compilation overhead on real hardware. As NISQ devices improve, absolute numbers will change, but relative trends are expected to hold.',
      items: [
        'Simulator-based results -- real hardware validation pending for some modules',
        'Noise models approximate; actual device noise may differ run to run',
        'Compilation and transpilation overhead not included in timing',
        'Results expected to improve with error-corrected quantum hardware',
      ],
    },
  ],
};

const sectionIcons: Record<string, React.ReactNode> = {
  'Quantum Simulation Environment': <Cpu className="w-5 h-5 text-[#00d4ff]" />,
  'Fairness Measures': <Scale className="w-5 h-5 text-[#00ff88]" />,
  'Metrics Collected': <BarChart3 className="w-5 h-5 text-purple-400" />,
  'Reproducibility': <GitBranch className="w-5 h-5 text-orange-400" />,
  'Statistical Validation': <Shield className="w-5 h-5 text-blue-400" />,
  'Limitations and Caveats': <BookOpen className="w-5 h-5 text-yellow-400" />,
};

export default function MethodologyPage() {
  const [data, setData] = useState<MethodologyData>(STATIC_METHODOLOGY);
  const [loading, setLoading] = useState(true);
  const [fromApi, setFromApi] = useState(false);

  useEffect(() => {
    const fetchMethodology = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/benchmark/methodology');
        if (response.ok) {
          const apiData = await response.json();
          setData(apiData);
          setFromApi(true);
        }
      } catch {
        // API unavailable -- keep static content
      } finally {
        setLoading(false);
      }
    };

    fetchMethodology();
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <QuantumParticles />
      <GlassNavigation />

      <main className="relative z-10 container mx-auto px-6 pt-28 pb-20 max-w-5xl">
        {/* Back Link */}
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <Link
            href="/showcase"
            className="inline-flex items-center gap-2 text-[#888888] hover:text-white transition text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Showcase
          </Link>
        </motion.div>

        {/* Header */}
        <motion.div
          className="mb-14"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#141414] border border-[#222222] mb-6">
            <FlaskConical className="w-4 h-4 text-[#00d4ff]" />
            <span className="text-[#888888] text-sm font-medium">
              {fromApi ? 'Live from API' : 'Reference Documentation'}
            </span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white via-blue-200 to-cyan-200 bg-clip-text text-transparent">
            {data.title}
          </h1>

          <p className="text-lg text-[#888888] leading-relaxed max-w-3xl">
            {data.overview}
          </p>
        </motion.div>

        {/* Loading skeleton */}
        {loading && (
          <div className="flex items-center gap-3 text-[#888888] mb-10">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span className="text-sm">Checking for live methodology data...</span>
          </div>
        )}

        {/* Sections */}
        <div className="space-y-8">
          {data.sections.map((section, index) => (
            <motion.div
              key={section.title}
              className="bg-[#141414]/80 backdrop-blur-sm border border-[#222222] rounded-xl p-6 md:p-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + index * 0.08 }}
            >
              <div className="flex items-center gap-3 mb-4">
                {sectionIcons[section.title] ?? <Atom className="w-5 h-5 text-[#00d4ff]" />}
                <h2 className="text-xl md:text-2xl font-bold">{section.title}</h2>
              </div>

              <p className="text-[#888888] leading-relaxed mb-5">{section.content}</p>

              {section.items && section.items.length > 0 && (
                <ul className="space-y-3">
                  {section.items.map((item, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <CheckCircle className="w-4 h-4 text-[#00ff88] flex-shrink-0 mt-1" />
                      <span className="text-[#e0e0e0] text-sm leading-relaxed">{item}</span>
                    </li>
                  ))}
                </ul>
              )}
            </motion.div>
          ))}
        </div>

        {/* API Badge */}
        <motion.div
          className="mt-12 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#141414] border border-[#222222] text-xs text-[#888888]">
            <div
              className={`w-2 h-2 rounded-full ${fromApi ? 'bg-[#00ff88]' : 'bg-yellow-500'}`}
            />
            {fromApi
              ? 'Methodology loaded from live API'
              : 'Showing static reference content (API not connected)'}
          </div>
        </motion.div>
      </main>
    </div>
  );
}
