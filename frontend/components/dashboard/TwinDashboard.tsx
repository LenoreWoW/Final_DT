'use client';

import React, { useState } from 'react';
import { Play, Pause, RefreshCw, BarChart2, Activity, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { twinService, Twin, SimulationResult } from '@/lib/api';

interface TwinDashboardProps {
  twinId: string;
  initialTwin?: Twin;
}

export function TwinDashboard({ twinId, initialTwin }: TwinDashboardProps) {
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [twin, setTwin] = useState<Twin | undefined>(initialTwin);

  const handleSimulate = async () => {
    setIsSimulating(true);
    try {
      const result = await twinService.runSimulation(twinId);
      setSimulationResult(result);
      // Refresh twin state
      const updatedTwin = await twinService.getTwin(twinId);
      setTwin(updatedTwin);
    } catch (error) {
      console.error('Simulation failed:', error);
    } finally {
      setIsSimulating(false);
    }
  };

  // Prepare chart data from simulation results
  const chartData = simulationResult?.results?.scenarios?.map((s: any) => ({
    name: `Scenario ${s.id}`,
    outcome: s.outcome,
    time: s.time_to_outcome,
  })).slice(0, 20) || [];

  return (
    <div className="space-y-6">
      {/* Twin Status Card */}
      <div className="bg-white border rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-800">{twin?.name || 'Loading...'}</h2>
            <p className="text-slate-500 text-sm mt-1">{twin?.description}</p>
          </div>
          <div className="flex gap-2">
            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium border border-green-200">
              {twin?.status?.toUpperCase()}
            </span>
            {twin?.domain && (
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium border border-blue-200">
                {twin.domain.toUpperCase()}
              </span>
            )}
          </div>
        </div>

        {/* Quantum Metrics */}
        {twin?.quantum_metrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <MetricCard 
              label="Qubits Allocated" 
              value={twin.quantum_metrics.qubits_allocated || 0} 
              icon={<CpuIcon />}
            />
            <MetricCard 
              label="Circuit Depth" 
              value={twin.quantum_metrics.circuit_depth || 0} 
              icon={<Activity className="w-4 h-4 text-orange-500" />}
            />
            <MetricCard 
              label="Entanglement Pairs" 
              value={twin.quantum_metrics.entanglement_pairs || 0} 
              icon={<Zap className="w-4 h-4 text-purple-500" />}
            />
            <MetricCard 
              label="Algorithm" 
              value={twin.quantum_metrics.primary_algorithm || 'N/A'} 
              icon={<BarChart2 className="w-4 h-4 text-blue-500" />}
            />
          </div>
        )}
      </div>

      {/* Simulation Control */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white border rounded-xl p-6 shadow-sm min-h-[400px]">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-600" />
              Simulation Results
            </h3>
            <button
              onClick={handleSimulate}
              disabled={isSimulating}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              {isSimulating ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isSimulating ? 'Simulating...' : 'Run Simulation'}
            </button>
          </div>

          {simulationResult ? (
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" hide />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="outcome" 
                    stroke="#4f46e5" 
                    strokeWidth={2}
                    dot={{ r: 4, fill: '#4f46e5' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-[300px] flex flex-col items-center justify-center text-slate-400 border-2 border-dashed rounded-lg">
              <BarChart2 className="w-12 h-12 mb-2 opacity-50" />
              <p>Run a simulation to see quantum states</p>
            </div>
          )}
        </div>

        {/* Quantum Advantage Panel */}
        <div className="bg-slate-900 text-white border border-slate-800 rounded-xl p-6 shadow-sm flex flex-col">
          <h3 className="font-semibold text-slate-100 mb-6 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Quantum Advantage
          </h3>
          
          {simulationResult ? (
            <div className="space-y-6 flex-1">
              <div>
                <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Speedup Factor</p>
                <div className="text-4xl font-bold text-green-400">
                  {simulationResult.quantum_advantage.speedup}x
                </div>
              </div>
              
              <div>
                <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Scenarios Tested</p>
                <div className="text-2xl font-semibold">
                  {simulationResult.results.statistics?.scenarios_run || 1000}+
                </div>
                <p className="text-xs text-slate-500 mt-1">Simultaneously via superposition</p>
              </div>

              <div>
                <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Classical Time</p>
                <div className="text-xl font-mono text-orange-300">
                  {simulationResult.quantum_advantage.classical_equivalent_seconds.toFixed(2)}s
                </div>
              </div>

              <div className="mt-auto pt-6 border-t border-slate-800">
                <p className="text-sm text-slate-400">
                  This simulation utilized quantum interference to filter suboptimal paths exponentially faster than classical methods.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
              Waiting for simulation data...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon }: { label: string, value: string | number, icon: React.ReactNode }) {
  return (
    <div className="p-4 bg-slate-50 rounded-lg border">
      <div className="flex items-center gap-2 mb-2 text-slate-500 text-xs font-medium uppercase tracking-wider">
        {icon}
        {label}
      </div>
      <div className="text-lg font-semibold text-slate-800">
        {value}
      </div>
    </div>
  );
}

function CpuIcon() {
  return (
    <svg className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  );
}

