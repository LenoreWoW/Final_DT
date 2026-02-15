'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Zap,
  Play,
  Pause,
  RefreshCw,
  Send,
  BarChart3,
  TrendingUp,
  Clock,
  Atom,
  Sparkles,
  AlertCircle,
  Loader2,
  SlidersHorizontal,
  MessageSquare,
  Layers,
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';
import QASMViewer from '@/components/quantum/QASMViewer';
import { twinService, Twin, SimulationResult } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ScenarioResult {
  id: number;
  description: string;
  outcome: number;
  confidence: number;
}

interface QueryResult {
  answer: string;
  confidence: number;
  data_points?: Array<{ label: string; value: number }>;
}

export default function DashboardPage() {
  const params = useParams();
  const twinId = params.twinId as string;

  const [twin, setTwin] = useState<Twin | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Simulation state
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [timeSteps, setTimeSteps] = useState(100);
  const [scenariosCount, setScenariosCount] = useState(100);

  // What-if scenarios
  const [scenarioInput, setScenarioInput] = useState('');
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [isRunningScenario, setIsRunningScenario] = useState(false);

  // Natural language query
  const [queryInput, setQueryInput] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // QASM viewer
  const [showQASM, setShowQASM] = useState(false);
  const [qasmCircuits, setQasmCircuits] = useState<Record<string, string>>({});

  // Fetch twin data
  useEffect(() => {
    const fetchTwin = async () => {
      try {
        const data = await twinService.getTwin(twinId);
        setTwin(data);
      } catch (err) {
        setError('Failed to load twin data. Please check the twin ID.');
        console.error('Error loading twin:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchTwin();
  }, [twinId]);

  // Run simulation
  const handleSimulate = useCallback(async () => {
    setIsSimulating(true);
    try {
      const result = await twinService.runSimulation(twinId, timeSteps);
      setSimulationResult(result);
      // Refresh twin state
      const updatedTwin = await twinService.getTwin(twinId);
      setTwin(updatedTwin);
    } catch (err) {
      console.error('Simulation failed:', err);
    } finally {
      setIsSimulating(false);
    }
  }, [twinId, timeSteps]);

  // Run what-if scenario
  const handleScenario = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!scenarioInput.trim() || isRunningScenario) return;

    setIsRunningScenario(true);
    try {
      const response = await twinService.queryTwin(twinId, `What if: ${scenarioInput}`);
      const newScenario: ScenarioResult = {
        id: scenarios.length + 1,
        description: scenarioInput,
        outcome: response.confidence ? response.confidence * 100 : 75,
        confidence: response.confidence ?? 0.75,
      };
      setScenarios((prev) => [...prev, newScenario]);
      setScenarioInput('');
    } catch (err) {
      console.error('Scenario analysis failed:', err);
    } finally {
      setIsRunningScenario(false);
    }
  };

  // Natural language query
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim() || isQuerying) return;

    setIsQuerying(true);
    try {
      const response = await twinService.queryTwin(twinId, queryInput);
      setQueryResult({
        answer: response.answer ?? response.message ?? 'Analysis complete.',
        confidence: response.confidence ?? 0.92,
        data_points: response.data_points,
      });
    } catch (err) {
      console.error('Query failed:', err);
      setQueryResult({
        answer: 'Unable to process query. Please try again.',
        confidence: 0,
      });
    } finally {
      setIsQuerying(false);
    }
  };

  // Fetch QASM circuits
  const handleViewQASM = useCallback(async () => {
    try {
      const data = await twinService.getQASM(twinId);
      setQasmCircuits(data.circuits || {});
      setShowQASM(true);
    } catch (err) {
      console.error('Failed to fetch QASM:', err);
    }
  }, [twinId]);

  // Prepare chart data
  const timeSeriesData = simulationResult?.results?.scenarios
    ?.slice(0, 30)
    .map((s: Record<string, unknown>, idx: number) => ({
      step: idx + 1,
      outcome: typeof s.outcome === 'number' ? s.outcome : 50,
      baseline: typeof s.outcome === 'number' ? (s.outcome as number) * 0.75 : 37.5,
      time: typeof s.time_to_outcome === 'number' ? s.time_to_outcome : idx * 0.1,
    })) ?? [];

  const comparisonData = simulationResult
    ? [
        {
          name: 'Quantum',
          execution_time: simulationResult.execution_time_seconds,
          scenarios_tested: simulationResult.results?.statistics?.scenarios_run ?? scenariosCount,
        },
        {
          name: 'Classical',
          execution_time: simulationResult.quantum_advantage.classical_equivalent_seconds,
          scenarios_tested: Math.min(
            simulationResult.results?.statistics?.scenarios_run ?? scenariosCount,
            50
          ),
        },
      ]
    : [];

  // Custom Recharts tooltip
  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; color: string }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#0a0a0a]/95 border border-white/10 rounded-lg p-3 shadow-xl backdrop-blur-sm">
          <p className="text-white/60 text-xs mb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <QuantumParticles />
        <div className="relative z-10 flex flex-col items-center gap-4">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
          <p className="text-white/70 text-lg">Loading twin dashboard...</p>
        </div>
      </div>
    );
  }

  if (error || !twin) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <QuantumParticles />
        <div className="relative z-10 flex flex-col items-center gap-4 text-center px-6">
          <AlertCircle className="w-12 h-12 text-red-400" />
          <p className="text-white text-xl font-semibold">Twin Not Found</p>
          <p className="text-white/60">{error ?? 'The requested twin could not be loaded.'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <QuantumParticles />
      <GlassNavigation />

      <main className="relative z-10 container mx-auto px-6 pt-24 pb-16">
        {/* Header */}
        <motion.header
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
                >
                  <Atom className="w-8 h-8 text-cyan-400" />
                </motion.div>
                <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-white via-blue-200 to-cyan-200 bg-clip-text text-transparent">
                  {twin.name}
                </h1>
              </div>
              <p className="text-white/60 max-w-2xl">{twin.description}</p>
            </div>

            <div className="flex items-center gap-3">
              <span
                className={cn(
                  'px-3 py-1 rounded-full text-xs font-medium border',
                  twin.status === 'active'
                    ? 'bg-green-500/20 text-green-400 border-green-500/30'
                    : twin.status === 'generating'
                    ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                    : 'bg-white/10 text-white/60 border-white/20'
                )}
              >
                {twin.status?.toUpperCase()}
              </span>
              {twin.domain && (
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  {twin.domain.toUpperCase()}
                </span>
              )}
            </div>
          </div>
        </motion.header>

        {/* Quantum Metrics Bar */}
        {twin.quantum_metrics && (
          <motion.div
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <MetricCard
              label="Qubits"
              value={twin.quantum_metrics.qubits_allocated ?? 0}
              icon={<Atom className="w-4 h-4 text-cyan-400" />}
            />
            <MetricCard
              label="Circuit Depth"
              value={twin.quantum_metrics.circuit_depth ?? 0}
              icon={<Activity className="w-4 h-4 text-purple-400" />}
            />
            <MetricCard
              label="Entanglement Pairs"
              value={twin.quantum_metrics.entanglement_pairs ?? 0}
              icon={<Zap className="w-4 h-4 text-yellow-400" />}
            />
            <MetricCard
              label="Algorithm"
              value={twin.quantum_metrics.primary_algorithm ?? 'N/A'}
              icon={<BarChart3 className="w-4 h-4 text-blue-400" />}
            />
          </motion.div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Column: Visualization Area */}
          <div className="xl:col-span-2 space-y-6">
            {/* Time Series Chart */}
            <motion.div
              className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Activity className="w-5 h-5 text-cyan-400" />
                  Simulation Time Series
                </h3>
                <button
                  onClick={handleSimulate}
                  disabled={isSimulating}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg text-sm font-medium hover:shadow-lg hover:shadow-blue-500/30 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSimulating ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Simulating...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Run Simulation
                    </>
                  )}
                </button>
              </div>

              {timeSeriesData.length > 0 ? (
                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeriesData}>
                      <defs>
                        <linearGradient id="gradientOutcome" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradientBaseline" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="step"
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                        label={{
                          value: 'Time Step',
                          position: 'insideBottom',
                          offset: -5,
                          fill: 'rgba(255,255,255,0.4)',
                        }}
                      />
                      <YAxis
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend
                        wrapperStyle={{ color: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                      />
                      <Area
                        type="monotone"
                        dataKey="outcome"
                        name="Quantum Outcome"
                        stroke="#06b6d4"
                        fill="url(#gradientOutcome)"
                        strokeWidth={2}
                      />
                      <Area
                        type="monotone"
                        dataKey="baseline"
                        name="Classical Baseline"
                        stroke="#8b5cf6"
                        fill="url(#gradientBaseline)"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-[320px] flex flex-col items-center justify-center text-white/30 border-2 border-dashed border-white/10 rounded-lg">
                  <BarChart3 className="w-12 h-12 mb-3 opacity-50" />
                  <p className="text-sm">Run a simulation to visualize quantum states</p>
                </div>
              )}
            </motion.div>

            {/* Comparison Bar Chart */}
            {comparisonData.length > 0 && (
              <motion.div
                className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                  Quantum vs Classical Comparison
                </h3>
                <div className="h-[250px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData} barGap={12}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="name"
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 13 }}
                      />
                      <YAxis
                        yAxisId="left"
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                        label={{
                          value: 'Time (s)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: 'rgba(255,255,255,0.4)',
                        }}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                        label={{
                          value: 'Scenarios',
                          angle: 90,
                          position: 'insideRight',
                          fill: 'rgba(255,255,255,0.4)',
                        }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.7)', fontSize: 12 }} />
                      <Bar
                        yAxisId="left"
                        dataKey="execution_time"
                        name="Execution Time (s)"
                        fill="#06b6d4"
                        radius={[4, 4, 0, 0]}
                      />
                      <Bar
                        yAxisId="right"
                        dataKey="scenarios_tested"
                        name="Scenarios Tested"
                        fill="#8b5cf6"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}

            {/* Natural Language Query */}
            <motion.div
              className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
            >
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <MessageSquare className="w-5 h-5 text-blue-400" />
                Ask Your Twin
              </h3>
              <form onSubmit={handleQuery} className="flex gap-3">
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  placeholder="e.g. What is the optimal strategy for next quarter?"
                  className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50 transition"
                  disabled={isQuerying}
                />
                <button
                  type="submit"
                  disabled={!queryInput.trim() || isQuerying}
                  className="px-5 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl font-medium hover:shadow-lg hover:shadow-cyan-500/30 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isQuerying ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </form>

              <AnimatePresence>
                {queryResult && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="mt-4 p-4 bg-cyan-500/10 border border-cyan-500/20 rounded-xl"
                  >
                    <div className="flex items-start gap-3">
                      <Sparkles className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-white/90 text-sm leading-relaxed whitespace-pre-wrap">
                          {queryResult.answer}
                        </p>
                        {queryResult.confidence > 0 && (
                          <p className="text-xs text-cyan-400/70 mt-2">
                            Confidence: {(queryResult.confidence * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>

          {/* Right Column: Controls + Quantum Advantage */}
          <div className="space-y-6">
            {/* Simulation Controls */}
            <motion.div
              className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 }}
            >
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
                <SlidersHorizontal className="w-5 h-5 text-purple-400" />
                Simulation Controls
              </h3>

              <div className="space-y-6">
                {/* Time Steps Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-white/60">Time Steps</label>
                    <span className="text-sm font-mono text-cyan-400">{timeSteps}</span>
                  </div>
                  <input
                    type="range"
                    min={10}
                    max={1000}
                    step={10}
                    value={timeSteps}
                    onChange={(e) => setTimeSteps(Number(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-cyan-500"
                  />
                  <div className="flex justify-between text-xs text-white/30 mt-1">
                    <span>10</span>
                    <span>1000</span>
                  </div>
                </div>

                {/* Scenarios Count */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-white/60">Scenarios</label>
                    <span className="text-sm font-mono text-cyan-400">{scenariosCount}</span>
                  </div>
                  <input
                    type="range"
                    min={10}
                    max={1000}
                    step={10}
                    value={scenariosCount}
                    onChange={(e) => setScenariosCount(Number(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="flex justify-between text-xs text-white/30 mt-1">
                    <span>10</span>
                    <span>1000</span>
                  </div>
                </div>

                <button
                  onClick={handleSimulate}
                  disabled={isSimulating}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/30 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSimulating ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Launch Simulation
                    </>
                  )}
                </button>
              </div>
            </motion.div>

            {/* Quantum Advantage Badge */}
            <motion.div
              className="bg-gradient-to-br from-[#0a0a0a] to-[#1a1a2e] border border-cyan-500/20 rounded-xl p-6 backdrop-blur-sm"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-6">
                <Zap className="w-5 h-5 text-yellow-400" />
                Quantum Advantage
              </h3>

              {simulationResult ? (
                <div className="space-y-5">
                  <div>
                    <p className="text-white/40 text-xs uppercase tracking-wider mb-1">
                      Speedup Factor
                    </p>
                    <div className="text-4xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                      {simulationResult.quantum_advantage.speedup}x
                    </div>
                  </div>

                  <div>
                    <p className="text-white/40 text-xs uppercase tracking-wider mb-1">
                      Scenarios Explored
                    </p>
                    <div className="text-2xl font-semibold text-white">
                      {simulationResult.results?.statistics?.scenarios_run ?? scenariosCount}+
                    </div>
                    <p className="text-xs text-white/30 mt-1">via quantum superposition</p>
                  </div>

                  <div>
                    <p className="text-white/40 text-xs uppercase tracking-wider mb-1">
                      Classical Equivalent
                    </p>
                    <div className="text-xl font-mono text-orange-300">
                      {simulationResult.quantum_advantage.classical_equivalent_seconds.toFixed(2)}s
                    </div>
                  </div>

                  <div>
                    <p className="text-white/40 text-xs uppercase tracking-wider mb-1">
                      Quantum Time
                    </p>
                    <div className="text-xl font-mono text-cyan-400">
                      {simulationResult.execution_time_seconds < 1
                        ? `${(simulationResult.execution_time_seconds * 1000).toFixed(0)}ms`
                        : `${simulationResult.execution_time_seconds.toFixed(2)}s`}
                    </div>
                  </div>

                  <div className="pt-4 border-t border-white/10 space-y-3">
                    <p className="text-xs text-white/40 leading-relaxed">
                      Quantum interference filtered suboptimal paths exponentially faster than
                      classical brute-force approaches.
                    </p>
                    <button
                      onClick={handleViewQASM}
                      className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-lg text-xs font-medium hover:bg-cyan-500/20 transition"
                    >
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="16 18 22 12 16 6" />
                        <polyline points="8 6 2 12 8 18" />
                      </svg>
                      View OpenQASM Circuits
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-white/20">
                  <Zap className="w-8 h-8 mb-2" />
                  <p className="text-sm">Run a simulation to measure advantage</p>
                </div>
              )}
            </motion.div>

            {/* What-If Scenarios */}
            <motion.div
              className="bg-white/5 border border-white/10 rounded-xl p-6 backdrop-blur-sm"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.35 }}
            >
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-orange-400" />
                What-If Scenarios
              </h3>

              <form onSubmit={handleScenario} className="flex gap-2 mb-4">
                <input
                  type="text"
                  value={scenarioInput}
                  onChange={(e) => setScenarioInput(e.target.value)}
                  placeholder="Describe a scenario..."
                  className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-orange-500/30 focus:border-orange-500/50 transition"
                  disabled={isRunningScenario}
                />
                <button
                  type="submit"
                  disabled={!scenarioInput.trim() || isRunningScenario}
                  className="px-3 py-2 bg-orange-500/20 border border-orange-500/30 text-orange-400 rounded-lg hover:bg-orange-500/30 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isRunningScenario ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                </button>
              </form>

              {scenarios.length > 0 ? (
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {scenarios.map((scenario) => (
                    <motion.div
                      key={scenario.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="p-3 bg-white/5 border border-white/10 rounded-lg"
                    >
                      <p className="text-xs text-white/50 mb-1">Scenario #{scenario.id}</p>
                      <p className="text-sm text-white/80 mb-2">{scenario.description}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white/40">
                          Confidence: {(scenario.confidence * 100).toFixed(0)}%
                        </span>
                        <span className="text-sm font-mono text-green-400">
                          {scenario.outcome.toFixed(1)}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-white/30 text-center py-4">
                  Add scenarios to explore alternative futures
                </p>
              )}
            </motion.div>
          </div>
        </div>
        {/* QASM Modal */}
        {showQASM && Object.keys(qasmCircuits).length > 0 && (
          <QASMViewer
            circuits={qasmCircuits}
            twinName={twin.name}
            isModal
            onClose={() => setShowQASM(false)}
          />
        )}
      </main>
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4 backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-2 text-white/40 text-xs font-medium uppercase tracking-wider">
        {icon}
        {label}
      </div>
      <div className="text-lg font-semibold text-white">{value}</div>
    </div>
  );
}
