'use client';

import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface ParticleFieldProps {
  count?: number;
  mousePosition: { x: number; y: number };
}

function ParticleField({ count = 2000, mousePosition }: ParticleFieldProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  // Generate random particles in 3D space
  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      // Position particles in a sphere
      const i3 = i * 3;
      const radius = 15 + Math.random() * 10;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);

      // Quantum colors - blue to cyan to purple gradient
      const colorChoice = Math.random();
      if (colorChoice < 0.33) {
        // Electric cyan
        colors[i3] = 0;
        colors[i3 + 1] = 0.831;
        colors[i3 + 2] = 1;
      } else if (colorChoice < 0.66) {
        // Quantum blue
        colors[i3] = 0.102;
        colors[i3 + 1] = 0.212;
        colors[i3 + 2] = 0.365;
      } else {
        // Neural purple
        colors[i3] = 0.486;
        colors[i3 + 1] = 0.227;
        colors[i3 + 2] = 0.929;
      }
    }

    return { positions, colors };
  }, [count]);

  // Animate particles
  useFrame((state) => {
    if (!pointsRef.current) return;

    const time = state.clock.getElapsedTime();

    // Smooth mouse following
    mouseRef.current.x += (mousePosition.x - mouseRef.current.x) * 0.05;
    mouseRef.current.y += (mousePosition.y - mouseRef.current.y) * 0.05;

    // Rotate entire particle field
    pointsRef.current.rotation.x = time * 0.05 + mouseRef.current.y * 0.5;
    pointsRef.current.rotation.y = time * 0.075 + mouseRef.current.x * 0.5;

    // Animate individual particles
    const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;

      // Add wave motion
      const x = positions[i3];
      const y = positions[i3 + 1];
      const z = positions[i3 + 2];

      positions[i3 + 1] += Math.sin(time + x * 0.1) * 0.01;
      positions[i3 + 2] += Math.cos(time + y * 0.1) * 0.01;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <Points ref={pointsRef} positions={particles.positions} colors={particles.colors}>
      <PointMaterial
        transparent
        vertexColors
        size={0.15}
        sizeAttenuation={true}
        depthWrite={false}
        opacity={0.8}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  );
}

interface QuantumParticlesProps {
  className?: string;
}

export default function QuantumParticles({ className = '' }: QuantumParticlesProps) {
  const [mousePosition, setMousePosition] = React.useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      setMousePosition({
        x: (event.clientX / window.innerWidth) * 2 - 1,
        y: -(event.clientY / window.innerHeight) * 2 + 1,
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className={`fixed inset-0 -z-10 ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 20], fov: 75 }}
        gl={{ antialias: true, alpha: true }}
      >
        <ParticleField mousePosition={mousePosition} />
      </Canvas>
    </div>
  );
}
