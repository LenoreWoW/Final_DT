'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface QuantumLoaderProps {
  fullScreen?: boolean;
  message?: string;
  progress?: number;
}

export default function QuantumLoader({
  fullScreen = false,
  message = 'Loading...',
  progress,
}: QuantumLoaderProps) {
  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col items-center justify-center z-50">
        <OrbitingParticles />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8 text-center"
        >
          <p className="text-white text-xl font-medium">{message}</p>
          {progress !== undefined && (
            <div className="mt-4 w-64">
              <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-white/60 text-sm mt-2">{progress}%</p>
            </div>
          )}
        </motion.div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <InlineOrbitingParticles />
      <span className="text-white/70">{message}</span>
    </div>
  );
}

function OrbitingParticles() {
  const particles = [
    { delay: 0, duration: 2, radius: 40, size: 8 },
    { delay: 0.33, duration: 2, radius: 40, size: 6 },
    { delay: 0.66, duration: 2, radius: 40, size: 10 },
  ];

  return (
    <div className="relative w-32 h-32">
      {/* Center atom */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="w-6 h-6 bg-gradient-to-br from-blue-400 to-purple-400 rounded-full blur-sm"
        />
      </div>

      {/* Orbiting particles */}
      {particles.map((particle, i) => (
        <motion.div
          key={i}
          className="absolute top-1/2 left-1/2"
          style={{
            width: particle.size,
            height: particle.size,
            marginLeft: -particle.size / 2,
            marginTop: -particle.size / 2,
          }}
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: particle.duration,
            delay: particle.delay,
            repeat: Infinity,
            ease: 'linear',
          }}
        >
          <motion.div
            className="w-full h-full rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 shadow-lg shadow-cyan-500/50"
            style={{
              translateX: particle.radius,
            }}
            animate={{
              scale: [1, 1.3, 1],
            }}
            transition={{
              duration: particle.duration / 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        </motion.div>
      ))}

      {/* Orbital rings */}
      <motion.div
        className="absolute inset-0 rounded-full border border-blue-500/30"
        animate={{
          rotate: 360,
          scale: [1, 1.05, 1],
        }}
        transition={{
          rotate: { duration: 3, repeat: Infinity, ease: 'linear' },
          scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
        }}
      />
      <motion.div
        className="absolute inset-2 rounded-full border border-purple-500/30"
        animate={{
          rotate: -360,
          scale: [1, 1.05, 1],
        }}
        transition={{
          rotate: { duration: 4, repeat: Infinity, ease: 'linear' },
          scale: { duration: 2.5, repeat: Infinity, ease: 'easeInOut' },
        }}
      />
    </div>
  );
}

function InlineOrbitingParticles() {
  return (
    <div className="relative w-6 h-6 flex items-center justify-center">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute w-1.5 h-1.5 bg-blue-400 rounded-full"
          animate={{
            rotate: 360,
            scale: [1, 1.5, 1],
          }}
          transition={{
            rotate: {
              duration: 2,
              delay: i * 0.33,
              repeat: Infinity,
              ease: 'linear',
            },
            scale: {
              duration: 1,
              delay: i * 0.33,
              repeat: Infinity,
              ease: 'easeInOut',
            },
          }}
          style={{
            transformOrigin: '50% 50%',
            translateX: 8,
          }}
        />
      ))}
    </div>
  );
}

// Export both components
export { OrbitingParticles, InlineOrbitingParticles };
