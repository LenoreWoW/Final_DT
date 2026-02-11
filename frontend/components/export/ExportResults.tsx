'use client';

import React, { useState } from 'react';
import { Download, FileText, FileJson, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

interface ExportResultsProps {
  twinData: any;
  twinName?: string;
}

export default function ExportResults({ twinData, twinName = 'quantum-twin' }: ExportResultsProps) {
  const [exporting, setExporting] = useState(false);

  const exportToJSON = () => {
    setExporting(true);
    try {
      const dataStr = JSON.stringify(twinData, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${twinName}-${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } finally {
      setTimeout(() => setExporting(false), 500);
    }
  };

  const exportToCSV = () => {
    setExporting(true);
    try {
      // Convert twin data to CSV format
      let csvContent = '';

      // If it's simulation results
      if (twinData.predictions && Array.isArray(twinData.predictions)) {
        csvContent = 'Timestamp,Value,Confidence\n';
        twinData.predictions.forEach((pred: any) => {
          csvContent += `${pred.timestamp || 'N/A'},${pred.value || 0},${pred.confidence || 0}\n`;
        });
      } else if (twinData.history) {
        // Time series data
        const keys = Object.keys(twinData.history);
        csvContent = keys.join(',') + '\n';

        const maxLength = Math.max(...keys.map(k => twinData.history[k].length));
        for (let i = 0; i < maxLength; i++) {
          const row = keys.map(k => twinData.history[k][i] || '');
          csvContent += row.join(',') + '\n';
        }
      } else {
        // Generic object to CSV
        csvContent = 'Key,Value\n';
        Object.entries(twinData).forEach(([key, value]) => {
          csvContent += `${key},"${typeof value === 'object' ? JSON.stringify(value) : value}"\n`;
        });
      }

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${twinName}-${Date.now()}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } finally {
      setTimeout(() => setExporting(false), 500);
    }
  };

  const exportToMarkdown = () => {
    setExporting(true);
    try {
      let mdContent = `# ${twinName}\n\n`;
      mdContent += `**Generated:** ${new Date().toISOString()}\n\n`;
      mdContent += `---\n\n`;

      // Twin metadata
      if (twinData.metadata) {
        mdContent += `## Metadata\n\n`;
        Object.entries(twinData.metadata).forEach(([key, value]) => {
          mdContent += `- **${key}**: ${value}\n`;
        });
        mdContent += `\n`;
      }

      // System information
      if (twinData.system) {
        mdContent += `## System Information\n\n`;
        mdContent += `- **Domain**: ${twinData.system.domain || 'Unknown'}\n`;
        mdContent += `- **Entities**: ${twinData.system.entities?.length || 0}\n`;
        mdContent += `- **Relationships**: ${twinData.system.relationships?.length || 0}\n`;
        mdContent += `\n`;
      }

      // Quantum algorithm
      if (twinData.algorithm) {
        mdContent += `## Quantum Algorithm\n\n`;
        mdContent += `- **Type**: ${twinData.algorithm.type || 'N/A'}\n`;
        mdContent += `- **Qubits**: ${twinData.algorithm.qubits || 'N/A'}\n`;
        mdContent += `- **Gates**: ${twinData.algorithm.gates || 'N/A'}\n`;
        mdContent += `\n`;
      }

      // Predictions/Results
      if (twinData.predictions) {
        mdContent += `## Predictions\n\n`;
        if (Array.isArray(twinData.predictions)) {
          mdContent += `| Timestamp | Value | Confidence |\n`;
          mdContent += `|-----------|-------|------------|\n`;
          twinData.predictions.slice(0, 10).forEach((pred: any) => {
            mdContent += `| ${pred.timestamp || 'N/A'} | ${pred.value || 0} | ${(pred.confidence * 100).toFixed(1)}% |\n`;
          });
        }
        mdContent += `\n`;
      }

      // Additional data
      mdContent += `## Full Data\n\n`;
      mdContent += `\`\`\`json\n${JSON.stringify(twinData, null, 2)}\n\`\`\`\n`;

      const blob = new Blob([mdContent], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${twinName}-${Date.now()}.md`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } finally {
      setTimeout(() => setExporting(false), 500);
    }
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-6">
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Download className="w-5 h-5 text-purple-400" />
        Export Results
      </h3>

      <p className="text-white/60 text-sm mb-6">
        Download your quantum twin data and simulation results in various formats
      </p>

      <div className="grid md:grid-cols-3 gap-4">
        {/* JSON Export */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={exportToJSON}
          disabled={exporting}
          className="flex flex-col items-center gap-3 p-6 bg-blue-500/10 border border-blue-500/30 rounded-xl hover:bg-blue-500/20 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {exporting ? (
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
          ) : (
            <FileJson className="w-8 h-8 text-blue-400" />
          )}
          <div className="text-center">
            <div className="font-semibold">JSON</div>
            <div className="text-xs text-white/60">Structured data</div>
          </div>
        </motion.button>

        {/* CSV Export */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={exportToCSV}
          disabled={exporting}
          className="flex flex-col items-center gap-3 p-6 bg-green-500/10 border border-green-500/30 rounded-xl hover:bg-green-500/20 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {exporting ? (
            <Loader2 className="w-8 h-8 text-green-400 animate-spin" />
          ) : (
            <FileText className="w-8 h-8 text-green-400" />
          )}
          <div className="text-center">
            <div className="font-semibold">CSV</div>
            <div className="text-xs text-white/60">Spreadsheet format</div>
          </div>
        </motion.button>

        {/* Markdown Report */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={exportToMarkdown}
          disabled={exporting}
          className="flex flex-col items-center gap-3 p-6 bg-purple-500/10 border border-purple-500/30 rounded-xl hover:bg-purple-500/20 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {exporting ? (
            <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
          ) : (
            <FileText className="w-8 h-8 text-purple-400" />
          )}
          <div className="text-center">
            <div className="font-semibold">Markdown</div>
            <div className="text-xs text-white/60">Documentation</div>
          </div>
        </motion.button>
      </div>

      <div className="mt-6 p-4 bg-white/5 rounded-lg">
        <p className="text-xs text-white/50">
          <strong>Note:</strong> Exported files contain complete twin state including metadata,
          quantum algorithm details, and all simulation results.
        </p>
      </div>
    </div>
  );
}
