'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ChatInterface } from '@/components/conversation/ChatInterface';
import { TwinDashboard } from '@/components/dashboard/TwinDashboard';
import FileUpload from '@/components/data/FileUpload';
import ExportResults from '@/components/export/ExportResults';
import CircuitVisualization from '@/components/quantum/CircuitVisualization';
import { twinService, Twin } from '@/lib/api';
import { createTwinSocket, WSMessage } from '@/lib/websocket';
import { Sparkles, Database, Download, Cpu, Check, Loader2 } from 'lucide-react';
import Link from 'next/link';
import QuantumParticles from '@/components/three/QuantumParticles';
import GlassNavigation from '@/components/layout/GlassNavigation';
import { motion, AnimatePresence } from 'framer-motion';

const GENERATION_STEPS = [
  { key: 'extraction', label: 'System Extracted', description: 'Analyzing your description...' },
  { key: 'entities', label: 'Entities Mapped', description: 'Identifying system components...' },
  { key: 'circuits', label: 'Circuits Building', description: 'Constructing quantum circuits...' },
  { key: 'optimizing', label: 'Optimizing', description: 'Optimizing circuit parameters...' },
  { key: 'validating', label: 'Validating', description: 'Verifying quantum advantage...' },
  { key: 'complete', label: 'Complete', description: 'Twin ready for simulation!' },
];

export default function BuilderPage() {
  const router = useRouter();
  const [activeTwinId, setActiveTwinId] = useState<string | null>(null);
  const [twinStatus, setTwinStatus] = useState<string>('draft');
  const [twinData, setTwinData] = useState<Twin | null>(null);
  const [uploadedData, setUploadedData] = useState<{ data: unknown; metadata: { rows: number; columns: number } } | null>(null);

  // Generation progress
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState(0);
  const [generationProgress, setGenerationProgress] = useState(0);

  const handleFileProcessed = (data: unknown, metadata: { rows: number; columns: number }) => {
    setUploadedData({ data, metadata });
  };

  const handleTwinCreated = (twinId: string) => {
    setActiveTwinId(twinId);
    setIsGenerating(true);
    setGenerationStep(0);
    setGenerationProgress(0);

    // Fallback progress timer (cancelled when WebSocket provides real updates)
    let wsConnected = false;
    const stepInterval = setInterval(() => {
      if (wsConnected) return;
      setGenerationStep((prev) => {
        const next = prev + 1;
        if (next >= GENERATION_STEPS.length) {
          clearInterval(stepInterval);
          setIsGenerating(false);
          twinService.getTwin(twinId).then(setTwinData);
          return prev;
        }
        setGenerationProgress((next / GENERATION_STEPS.length) * 100);
        return next;
      });
    }, 800);

    // Connect WebSocket for real-time updates
    const { cleanup } = createTwinSocket(twinId, (msg: WSMessage) => {
      if (msg.type === 'generation_progress' && typeof msg.progress === 'number') {
        wsConnected = true;
        clearInterval(stepInterval);
        setGenerationProgress(msg.progress * 100);
        const stepIdx = GENERATION_STEPS.findIndex((s) => s.key === msg.step);
        if (stepIdx >= 0) setGenerationStep(stepIdx);
      }
      if (msg.type === 'simulation_complete' || (msg.type === 'generation_progress' && msg.step === 'complete')) {
        setIsGenerating(false);
        clearInterval(stepInterval);
        twinService.getTwin(twinId).then(setTwinData);
      }
    });

    // Cleanup WS on unmount
    return () => cleanup();
  };

  // Redirect to dashboard once twin is ready and user interacts
  const handleGoToDashboard = () => {
    if (activeTwinId) {
      router.push(`/dashboard/${activeTwinId}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* WebGL Particle Background */}
      <QuantumParticles />

      {/* Glass Navigation */}
      <GlassNavigation />

      <main className="container mx-auto px-6 py-8 pt-24">
        <motion.header
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
            >
              <Sparkles className="w-8 h-8 text-blue-400" />
            </motion.div>
            <h1 className="text-4xl font-bold text-white">Universal Twin Builder</h1>
          </div>
          <p className="text-lg text-white/70">
            Describe any system to generate a quantum-powered digital twin.
          </p>
        </motion.header>

        {/* Tabs for different sections */}
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex gap-2 bg-white/5 p-1 rounded-lg inline-flex border border-white/10">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-md font-medium shadow-lg shadow-blue-500/30"
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Build Twin
              </div>
            </motion.button>
            {activeTwinId && twinData && (
              <>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-2 text-white/70 hover:text-white hover:bg-white/10 rounded-md transition"
                >
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Circuit
                  </div>
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-2 text-white/70 hover:text-white hover:bg-white/10 rounded-md transition"
                >
                  <div className="flex items-center gap-2">
                    <Download className="w-4 h-4" />
                    Export
                  </div>
                </motion.button>
              </>
            )}
          </div>
        </motion.div>

        {/* Generation Progress Stepper */}
        <AnimatePresence>
          {isGenerating && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <div className="bg-white/5 border border-cyan-500/20 rounded-xl p-6 backdrop-blur-sm">
                <div className="flex items-center gap-2 mb-4">
                  <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
                  <h3 className="text-lg font-semibold text-white">Generating Quantum Twin...</h3>
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 bg-white/10 rounded-full mb-6 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${generationProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>

                {/* Steps */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                  {GENERATION_STEPS.map((step, idx) => {
                    const isComplete = idx < generationStep;
                    const isCurrent = idx === generationStep;

                    return (
                      <div
                        key={step.key}
                        className={`flex flex-col items-center text-center p-3 rounded-lg transition-all ${
                          isComplete
                            ? 'bg-green-500/10 border border-green-500/30'
                            : isCurrent
                            ? 'bg-cyan-500/10 border border-cyan-500/30'
                            : 'bg-white/5 border border-white/10'
                        }`}
                      >
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center mb-2 ${
                            isComplete
                              ? 'bg-green-500/20 text-green-400'
                              : isCurrent
                              ? 'bg-cyan-500/20 text-cyan-400'
                              : 'bg-white/10 text-white/30'
                          }`}
                        >
                          {isComplete ? (
                            <Check className="w-4 h-4" />
                          ) : isCurrent ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <span className="text-xs">{idx + 1}</span>
                          )}
                        </div>
                        <p
                          className={`text-xs font-medium ${
                            isComplete
                              ? 'text-green-400'
                              : isCurrent
                              ? 'text-cyan-400'
                              : 'text-white/30'
                          }`}
                        >
                          {step.label}
                        </p>
                      </div>
                    );
                  })}
                </div>

                <p className="text-xs text-white/40 mt-4 text-center">
                  {GENERATION_STEPS[generationStep]?.description}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column: Chat & Data Upload */}
          <div className="xl:col-span-1 space-y-6">
            <ChatInterface
              onTwinCreated={handleTwinCreated}
              onStatusChange={setTwinStatus}
            />

            {/* Data Upload Section */}
            <div>
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-400" />
                Data Upload (Optional)
              </h3>
              <FileUpload onFileProcessed={handleFileProcessed} />
              {uploadedData && (
                <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <p className="text-sm text-blue-200">
                    Data loaded: {uploadedData.metadata.rows} rows x {uploadedData.metadata.columns} columns
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Dashboard & Advanced Features */}
          <div className="xl:col-span-2 space-y-6">
            {activeTwinId ? (
              <>
                <TwinDashboard twinId={activeTwinId} />

                {/* Go to full dashboard button */}
                {twinData && !isGenerating && (
                  <motion.button
                    onClick={handleGoToDashboard}
                    className="w-full py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl font-medium text-white hover:shadow-lg hover:shadow-cyan-500/30 transition flex items-center justify-center gap-2"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    Open Full Dashboard
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                  </motion.button>
                )}

                {/* Circuit Visualization */}
                {twinData?.algorithm && (
                  <CircuitVisualization
                    numQubits={twinData.algorithm.qubits || 4}
                    algorithm={twinData.algorithm.type || 'QAOA'}
                  />
                )}

                {/* Export Section */}
                {twinData && (
                  <ExportResults
                    twinData={twinData}
                    twinName={twinData.name || 'quantum-twin'}
                  />
                )}
              </>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 }}
                className="h-[600px] border-2 border-dashed border-white/20 rounded-xl flex flex-col items-center justify-center text-white/40 bg-white/5 backdrop-blur-sm"
              >
                <motion.div
                  animate={{
                    scale: [1, 1.1, 1],
                    rotate: [0, 180, 360],
                  }}
                  transition={{
                    duration: 4,
                    repeat: Infinity,
                    ease: 'linear',
                  }}
                  className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center mb-4 border border-blue-500/30"
                >
                  <Sparkles className="w-8 h-8 text-blue-400" />
                </motion.div>
                <p className="text-lg font-medium text-white/60">No Active Twin</p>
                <p className="text-sm">Start chatting to build your quantum digital twin</p>
              </motion.div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
