'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Atom, Zap, GitBranch } from 'lucide-react';

interface Gate {
  type: 'H' | 'X' | 'Y' | 'Z' | 'CNOT' | 'RX' | 'RY' | 'RZ' | 'CZ' | 'SWAP';
  qubit: number;
  target?: number;
  parameter?: number;
  position: number;
}

interface CircuitVisualizationProps {
  numQubits: number;
  gates?: Gate[];
  algorithm?: string;
  showLabels?: boolean;
}

export default function CircuitVisualization({
  numQubits = 4,
  gates = [],
  algorithm = 'QAOA',
  showLabels = true
}: CircuitVisualizationProps) {
  // Generate example gates if none provided
  const displayGates = gates.length > 0 ? gates : generateExampleGates(numQubits, algorithm);

  const maxPosition = Math.max(...displayGates.map(g => g.position)) + 1;
  const gridWidth = maxPosition * 80 + 100;
  const gridHeight = numQubits * 80 + 40;

  function generateExampleGates(qubits: number, algo: string): Gate[] {
    const exampleGates: Gate[] = [];

    if (algo === 'QAOA') {
      // QAOA pattern: Hadamards, then parameterized gates, then CNOTs
      for (let i = 0; i < qubits; i++) {
        exampleGates.push({ type: 'H', qubit: i, position: 0 });
        exampleGates.push({ type: 'RZ', qubit: i, parameter: Math.PI / 4, position: 1 });
      }
      for (let i = 0; i < qubits - 1; i++) {
        exampleGates.push({ type: 'CNOT', qubit: i, target: i + 1, position: 2 });
      }
      for (let i = 0; i < qubits; i++) {
        exampleGates.push({ type: 'RX', qubit: i, parameter: Math.PI / 3, position: 3 });
      }
    } else if (algo === 'VQE') {
      // VQE pattern
      for (let i = 0; i < qubits; i++) {
        exampleGates.push({ type: 'RY', qubit: i, parameter: Math.PI / 6, position: 0 });
      }
      for (let i = 0; i < qubits - 1; i += 2) {
        exampleGates.push({ type: 'CNOT', qubit: i, target: i + 1, position: 1 });
      }
      for (let i = 1; i < qubits - 1; i += 2) {
        exampleGates.push({ type: 'CNOT', qubit: i, target: i + 1, position: 2 });
      }
    } else {
      // Generic quantum circuit
      for (let i = 0; i < qubits; i++) {
        exampleGates.push({ type: 'H', qubit: i, position: 0 });
      }
      for (let i = 0; i < qubits - 1; i++) {
        exampleGates.push({ type: 'CNOT', qubit: i, target: i + 1, position: 1 });
      }
    }

    return exampleGates;
  }

  const getGateColor = (type: string) => {
    switch (type) {
      case 'H': return '#3b82f6'; // blue
      case 'X': return '#ef4444'; // red
      case 'Y': return '#f59e0b'; // amber
      case 'Z': return '#8b5cf6'; // purple
      case 'CNOT': return '#10b981'; // green
      case 'RX': return '#ec4899'; // pink
      case 'RY': return '#06b6d4'; // cyan
      case 'RZ': return '#a855f7'; // purple
      case 'CZ': return '#14b8a6'; // teal
      case 'SWAP': return '#f97316'; // orange
      default: return '#6b7280'; // gray
    }
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Atom className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h3 className="text-xl font-bold">Quantum Circuit</h3>
            {showLabels && (
              <p className="text-sm text-white/60">{algorithm} Algorithm • {numQubits} Qubits</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm text-white/60">
          <Zap className="w-4 h-4" />
          <span>{displayGates.length} Gates</span>
        </div>
      </div>

      {/* Circuit Diagram */}
      <div className="overflow-x-auto">
        <svg
          width={gridWidth}
          height={gridHeight}
          className="mx-auto"
          style={{ minWidth: '600px' }}
        >
          {/* Qubit lines */}
          {Array.from({ length: numQubits }).map((_, i) => {
            const y = 40 + i * 80;
            return (
              <g key={`qubit-${i}`}>
                {/* Qubit label */}
                <text
                  x="10"
                  y={y + 5}
                  className="fill-white/70 text-sm font-mono"
                >
                  q{i}
                </text>

                {/* Qubit line */}
                <line
                  x1="50"
                  y1={y}
                  x2={gridWidth - 20}
                  y2={y}
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="2"
                />

                {/* Initial state */}
                <text
                  x="20"
                  y={y + 25}
                  className="fill-white/40 text-xs font-mono"
                >
                  |0⟩
                </text>
              </g>
            );
          })}

          {/* Gates */}
          {displayGates.map((gate, idx) => {
            const x = 100 + gate.position * 80;
            const y = 40 + gate.qubit * 80;
            const color = getGateColor(gate.type);

            if (gate.type === 'CNOT' && gate.target !== undefined) {
              const targetY = 40 + gate.target * 80;
              return (
                <g key={`gate-${idx}`}>
                  {/* Control to target line */}
                  <motion.line
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ delay: idx * 0.05, duration: 0.3 }}
                    x1={x}
                    y1={y}
                    x2={x}
                    y2={targetY}
                    stroke={color}
                    strokeWidth="3"
                  />

                  {/* Control qubit (filled circle) */}
                  <motion.circle
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: idx * 0.05, type: 'spring' }}
                    cx={x}
                    cy={y}
                    r="8"
                    fill={color}
                  />

                  {/* Target qubit (circle with X) */}
                  <motion.g
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: idx * 0.05, type: 'spring' }}
                  >
                    <circle
                      cx={x}
                      cy={targetY}
                      r="20"
                      fill="none"
                      stroke={color}
                      strokeWidth="3"
                    />
                    <line
                      x1={x - 12}
                      y1={targetY - 12}
                      x2={x + 12}
                      y2={targetY + 12}
                      stroke={color}
                      strokeWidth="3"
                    />
                    <line
                      x1={x - 12}
                      y1={targetY + 12}
                      x2={x + 12}
                      y2={targetY - 12}
                      stroke={color}
                      strokeWidth="3"
                    />
                  </motion.g>
                </g>
              );
            } else {
              // Single qubit gate
              return (
                <motion.g
                  key={`gate-${idx}`}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: idx * 0.05, type: 'spring' }}
                >
                  <rect
                    x={x - 20}
                    y={y - 20}
                    width="40"
                    height="40"
                    fill={color}
                    rx="6"
                  />
                  <text
                    x={x}
                    y={y + 6}
                    className="fill-white text-sm font-bold"
                    textAnchor="middle"
                  >
                    {gate.type}
                  </text>
                  {gate.parameter && (
                    <text
                      x={x}
                      y={y + 35}
                      className="fill-white/60 text-xs"
                      textAnchor="middle"
                    >
                      {(gate.parameter / Math.PI).toFixed(2)}π
                    </text>
                  )}
                </motion.g>
              );
            }
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-6 border-t border-white/10">
        <h4 className="text-sm font-semibold text-white/70 mb-3">Gate Legend</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { type: 'H', name: 'Hadamard', desc: 'Superposition' },
            { type: 'X', name: 'Pauli-X', desc: 'Bit flip' },
            { type: 'CNOT', name: 'CNOT', desc: 'Entanglement' },
            { type: 'RZ', name: 'RZ(θ)', desc: 'Z-rotation' },
          ].map(gate => (
            <div key={gate.type} className="flex items-center gap-2">
              <div
                className="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold"
                style={{ backgroundColor: getGateColor(gate.type) }}
              >
                {gate.type}
              </div>
              <div className="text-xs">
                <div className="text-white/80 font-semibold">{gate.name}</div>
                <div className="text-white/50">{gate.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Circuit Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-2xl font-bold text-blue-400">{numQubits}</div>
          <div className="text-xs text-white/60">Qubits</div>
        </div>
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-400">{displayGates.length}</div>
          <div className="text-xs text-white/60">Gates</div>
        </div>
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-2xl font-bold text-green-400">{maxPosition}</div>
          <div className="text-xs text-white/60">Depth</div>
        </div>
      </div>
    </div>
  );
}
