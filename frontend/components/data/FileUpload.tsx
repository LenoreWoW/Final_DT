'use client';

import React, { useState, useCallback } from 'react';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface FileUploadProps {
  onFileProcessed: (data: any, metadata: any) => void;
  acceptedFormats?: string[];
  maxSizeMB?: number;
}

interface ParsedData {
  headers: string[];
  rows: any[][];
  preview: any[];
  schema: {
    [key: string]: {
      type: string;
      sample: any;
      unique_count?: number;
      min?: number;
      max?: number;
    };
  };
}

export default function FileUpload({
  onFileProcessed,
  acceptedFormats = ['.csv', '.json', '.xlsx', '.txt'],
  maxSizeMB = 10
}: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<ParsedData | null>(null);

  const parseCSV = (text: string): ParsedData => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = lines.slice(1).map(line =>
      line.split(',').map(cell => cell.trim())
    );

    // Detect schema
    const schema: any = {};
    headers.forEach((header, idx) => {
      const column = rows.map(row => row[idx]);
      const nonEmpty = column.filter(v => v !== '' && v !== null);

      // Try to infer type
      const numericCount = nonEmpty.filter(v => !isNaN(Number(v))).length;
      const isNumeric = numericCount / nonEmpty.length > 0.8;

      const unique = new Set(nonEmpty);

      schema[header] = {
        type: isNumeric ? 'numeric' : 'categorical',
        sample: nonEmpty[0],
        unique_count: unique.size,
        min: isNumeric ? Math.min(...nonEmpty.map(Number)) : undefined,
        max: isNumeric ? Math.max(...nonEmpty.map(Number)) : undefined,
      };
    });

    return {
      headers,
      rows,
      preview: rows.slice(0, 5),
      schema
    };
  };

  const parseJSON = (text: string): ParsedData => {
    const data = JSON.parse(text);

    if (Array.isArray(data)) {
      const headers = Object.keys(data[0] || {});
      const rows = data.map(obj => headers.map(h => obj[h]));

      const schema: any = {};
      headers.forEach(header => {
        const column = data.map((obj: any) => obj[header]);
        const nonEmpty = column.filter((v: any) => v !== '' && v !== null && v !== undefined);

        const type = typeof nonEmpty[0];
        const unique = new Set(nonEmpty);

        schema[header] = {
          type: type === 'number' ? 'numeric' : 'categorical',
          sample: nonEmpty[0],
          unique_count: unique.size
        };
      });

      return {
        headers,
        rows,
        preview: rows.slice(0, 5),
        schema
      };
    }

    throw new Error('JSON must be an array of objects');
  };

  const handleFile = useCallback(async (selectedFile: File) => {
    setError(null);
    setIsProcessing(true);

    try {
      // Check file size
      const sizeMB = selectedFile.size / (1024 * 1024);
      if (sizeMB > maxSizeMB) {
        throw new Error(`File size (${sizeMB.toFixed(1)}MB) exceeds maximum (${maxSizeMB}MB)`);
      }

      const text = await selectedFile.text();
      let parsed: ParsedData;

      if (selectedFile.name.endsWith('.csv')) {
        parsed = parseCSV(text);
      } else if (selectedFile.name.endsWith('.json')) {
        parsed = parseJSON(text);
      } else {
        throw new Error('Unsupported file format');
      }

      setParsedData(parsed);
      setFile(selectedFile);

      // Notify parent
      onFileProcessed(parsed, {
        filename: selectedFile.name,
        size: selectedFile.size,
        rows: parsed.rows.length,
        columns: parsed.headers.length
      });

    } catch (err: any) {
      setError(err.message);
      setFile(null);
      setParsedData(null);
    } finally {
      setIsProcessing(false);
    }
  }, [maxSizeMB, onFileProcessed]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFile(droppedFile);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFile(selectedFile);
    }
  }, [handleFile]);

  const removeFile = () => {
    setFile(null);
    setParsedData(null);
    setError(null);
  };

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      {!file && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            border-2 border-dashed rounded-xl p-8 text-center transition-all
            ${isDragging
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-white/20 hover:border-white/40 bg-white/5'
            }
          `}
        >
          <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragging ? 'text-blue-400' : 'text-white/60'}`} />

          <h3 className="text-lg font-semibold mb-2">
            {isDragging ? 'Drop file here' : 'Upload Data File'}
          </h3>

          <p className="text-white/60 text-sm mb-4">
            Drag and drop or click to select
          </p>

          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept={acceptedFormats.join(',')}
            onChange={handleFileInput}
            disabled={isProcessing}
          />

          <label
            htmlFor="file-upload"
            className="inline-block px-6 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg cursor-pointer transition"
          >
            Select File
          </label>

          <p className="text-xs text-white/40 mt-4">
            Supported: {acceptedFormats.join(', ')} (max {maxSizeMB}MB)
          </p>
        </div>
      )}

      {/* Processing State */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4"
          >
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
              <span className="text-blue-300">Processing file...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error State */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-500/10 border border-red-500/30 rounded-xl p-4"
          >
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-red-300 mb-1">Upload Error</h4>
                <p className="text-sm text-red-200">{error}</p>
              </div>
              <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
                <X className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Success State with Preview */}
      <AnimatePresence>
        {file && parsedData && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-green-500/10 border border-green-500/30 rounded-xl p-6"
          >
            {/* File Info */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-green-500/20 rounded-lg">
                  <File className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-green-300 flex items-center gap-2">
                    {file.name}
                    <CheckCircle className="w-4 h-4" />
                  </h4>
                  <p className="text-sm text-green-200/70 mt-1">
                    {parsedData.rows.length} rows × {parsedData.headers.length} columns
                    {' • '}
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                onClick={removeFile}
                className="p-2 hover:bg-white/10 rounded-lg transition"
              >
                <X className="w-5 h-5 text-white/60" />
              </button>
            </div>

            {/* Schema Info */}
            <div className="mb-4">
              <h5 className="text-sm font-semibold text-white/80 mb-2">Detected Schema:</h5>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {Object.entries(parsedData.schema).map(([col, info]) => (
                  <div key={col} className="bg-white/5 rounded-lg p-2">
                    <div className="text-sm font-mono text-white/90">{col}</div>
                    <div className="text-xs text-white/60">
                      {info.type}
                      {info.unique_count && ` (${info.unique_count} unique)`}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Data Preview */}
            <div>
              <h5 className="text-sm font-semibold text-white/80 mb-2">Preview (first 5 rows):</h5>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10">
                      {parsedData.headers.map((header, idx) => (
                        <th key={idx} className="text-left px-3 py-2 text-white/70 font-semibold">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parsedData.preview.map((row, rowIdx) => (
                      <tr key={rowIdx} className="border-b border-white/5">
                        {row.map((cell: string | number, cellIdx: number) => (
                          <td key={cellIdx} className="px-3 py-2 text-white/80 font-mono text-xs">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
