'use client';

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Download, X, Check, Code2, ChevronDown } from 'lucide-react';

interface QASMViewerProps {
  circuits: Record<string, string>;
  twinName?: string;
  onClose?: () => void;
  isModal?: boolean;
}

// Basic QASM syntax highlighting
function highlightQASM(code: string): React.ReactNode[] {
  return code.split('\n').map((line, idx) => {
    let highlighted = line;

    // Comments
    if (line.trim().startsWith('//')) {
      return (
        <div key={idx} className="text-white/30 italic">
          {line}
        </div>
      );
    }

    // Build spans for different token types
    const parts: React.ReactNode[] = [];
    let remaining = line;
    let partIdx = 0;

    // OPENQASM and include
    if (/^OPENQASM/.test(remaining) || /^include/.test(remaining)) {
      parts.push(
        <span key={partIdx++} className="text-purple-400 font-semibold">
          {remaining}
        </span>
      );
      return <div key={idx}>{parts}</div>;
    }

    // Register declarations
    const regMatch = remaining.match(/^(qreg|creg)\s+(\w+)\[(\d+)\];/);
    if (regMatch) {
      parts.push(
        <span key={partIdx++} className="text-blue-400 font-semibold">
          {regMatch[1]}
        </span>,
        <span key={partIdx++} className="text-white"> </span>,
        <span key={partIdx++} className="text-green-400">
          {regMatch[2]}
        </span>,
        <span key={partIdx++} className="text-yellow-400">
          [{regMatch[3]}]
        </span>,
        <span key={partIdx++} className="text-white/50">;</span>
      );
      return <div key={idx}>{parts}</div>;
    }

    // Measure
    const measureMatch = remaining.match(/^(measure)\s+(.*?)\s*->\s*(.*?);/);
    if (measureMatch) {
      parts.push(
        <span key={partIdx++} className="text-red-400 font-semibold">
          {measureMatch[1]}
        </span>,
        <span key={partIdx++} className="text-white"> </span>,
        <span key={partIdx++} className="text-cyan-400">
          {measureMatch[2]}
        </span>,
        <span key={partIdx++} className="text-white/50"> {'->'} </span>,
        <span key={partIdx++} className="text-orange-400">
          {measureMatch[3]}
        </span>,
        <span key={partIdx++} className="text-white/50">;</span>
      );
      return <div key={idx}>{parts}</div>;
    }

    // Gate operations
    const gateMatch = remaining.match(/^(\w+)(\(.*?\))?\s+(.*?);/);
    if (gateMatch) {
      const gateName = gateMatch[1];
      const gateParam = gateMatch[2] || '';
      const qubits = gateMatch[3];

      // Color gates by type
      let gateColor = 'text-cyan-400';
      if (['h', 'x', 'y', 'z', 's', 't'].includes(gateName))
        gateColor = 'text-cyan-400 font-semibold';
      else if (['cx', 'cz', 'swap'].includes(gateName))
        gateColor = 'text-yellow-400 font-semibold';
      else if (['rx', 'ry', 'rz'].includes(gateName))
        gateColor = 'text-green-400';
      else if (gateName === 'barrier') gateColor = 'text-white/30';

      parts.push(
        <span key={partIdx++} className={gateColor}>
          {gateName}
        </span>
      );
      if (gateParam) {
        parts.push(
          <span key={partIdx++} className="text-orange-300">
            {gateParam}
          </span>
        );
      }
      parts.push(
        <span key={partIdx++} className="text-white"> </span>,
        <span key={partIdx++} className="text-white/70">
          {qubits}
        </span>,
        <span key={partIdx++} className="text-white/50">;</span>
      );
      return <div key={idx}>{parts}</div>;
    }

    // Default: plain text
    return (
      <div key={idx} className="text-white/70">
        {line || '\u00A0'}
      </div>
    );
  });
}

export default function QASMViewer({
  circuits,
  twinName = 'circuit',
  onClose,
  isModal = false,
}: QASMViewerProps) {
  const circuitNames = Object.keys(circuits);
  const [activeCircuit, setActiveCircuit] = useState(circuitNames[0] || '');
  const [copied, setCopied] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const currentQASM = circuits[activeCircuit] || '';
  const lineCount = currentQASM.split('\n').length;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(currentQASM);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const ta = document.createElement('textarea');
      ta.value = currentQASM;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [currentQASM]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([currentQASM], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${twinName}_${activeCircuit}.qasm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [currentQASM, twinName, activeCircuit]);

  const content = (
    <div className="bg-[#0d1117] border border-white/10 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Code2 className="w-5 h-5 text-cyan-400" />
          <span className="text-sm font-semibold text-white">OpenQASM 2.0</span>

          {/* Circuit selector */}
          {circuitNames.length > 1 && (
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-1 px-3 py-1 bg-white/10 border border-white/20 rounded-md text-xs text-white/80 hover:bg-white/15 transition"
              >
                {activeCircuit.replace(/_/g, ' ')}
                <ChevronDown className="w-3 h-3" />
              </button>
              {dropdownOpen && (
                <div className="absolute top-full left-0 mt-1 bg-[#1a1a2e] border border-white/20 rounded-md shadow-xl z-50 min-w-[200px]">
                  {circuitNames.map((name) => (
                    <button
                      key={name}
                      onClick={() => {
                        setActiveCircuit(name);
                        setDropdownOpen(false);
                      }}
                      className={`block w-full text-left px-3 py-2 text-xs hover:bg-white/10 transition ${
                        name === activeCircuit
                          ? 'text-cyan-400 bg-cyan-500/10'
                          : 'text-white/70'
                      }`}
                    >
                      {name.replace(/_/g, ' ')}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <span className="text-xs text-white/30">{lineCount} lines</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-md text-xs text-white/80 transition"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-green-400" />
                <span className="text-green-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                Copy
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1 px-3 py-1.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-md text-xs text-white/80 transition"
          >
            <Download className="w-3 h-3" />
            .qasm
          </button>
          {isModal && onClose && (
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-white/10 rounded-md transition"
            >
              <X className="w-4 h-4 text-white/60" />
            </button>
          )}
        </div>
      </div>

      {/* Code Area */}
      <div className="relative overflow-auto max-h-[500px]">
        <div className="flex">
          {/* Line numbers */}
          <div className="flex-shrink-0 py-4 pr-3 pl-4 text-right select-none border-r border-white/5">
            {currentQASM.split('\n').map((_, idx) => (
              <div key={idx} className="text-xs text-white/20 leading-5 font-mono">
                {idx + 1}
              </div>
            ))}
          </div>

          {/* Code */}
          <pre className="flex-1 py-4 px-4 overflow-x-auto">
            <code className="text-xs leading-5 font-mono">
              {highlightQASM(currentQASM)}
            </code>
          </pre>
        </div>
      </div>

      {/* Footer stats */}
      <div className="flex items-center gap-4 px-4 py-2 bg-white/5 border-t border-white/10 text-xs text-white/30">
        <span>{circuitNames.length} circuit{circuitNames.length !== 1 ? 's' : ''}</span>
        <span>{lineCount} lines</span>
        <span>{currentQASM.length} chars</span>
      </div>
    </div>
  );

  if (isModal) {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
          onClick={(e) => {
            if (e.target === e.currentTarget && onClose) onClose();
          }}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="w-full max-w-3xl"
          >
            {content}
          </motion.div>
        </motion.div>
      </AnimatePresence>
    );
  }

  return content;
}
