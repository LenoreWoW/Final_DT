'use client';

import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface ScrollRevealProps {
  children: React.ReactNode;
  direction?: 'up' | 'down' | 'left' | 'right' | 'fade';
  delay?: number;
  duration?: number;
  className?: string;
}

export default function ScrollReveal({
  children,
  direction = 'up',
  delay = 0,
  duration = 1,
  className = '',
}: ScrollRevealProps) {
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!elementRef.current) return;

    const element = elementRef.current;

    // Define initial states based on direction
    const initialStates: Record<string, any> = {
      up: { y: 60, opacity: 0 },
      down: { y: -60, opacity: 0 },
      left: { x: 60, opacity: 0 },
      right: { x: -60, opacity: 0 },
      fade: { opacity: 0 },
    };

    gsap.fromTo(
      element,
      initialStates[direction],
      {
        y: 0,
        x: 0,
        opacity: 1,
        duration,
        delay,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: element,
          start: 'top 85%',
          end: 'bottom 15%',
          toggleActions: 'play none none reverse',
        },
      }
    );

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => {
        if (trigger.trigger === element) {
          trigger.kill();
        }
      });
    };
  }, [direction, delay, duration]);

  return (
    <div ref={elementRef} className={className}>
      {children}
    </div>
  );
}

// Staggered reveal for multiple children
interface StaggerRevealProps {
  children: React.ReactNode;
  stagger?: number;
  className?: string;
}

export function StaggerReveal({
  children,
  stagger = 0.1,
  className = '',
}: StaggerRevealProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const elements = containerRef.current.children;

    gsap.fromTo(
      elements,
      { y: 40, opacity: 0 },
      {
        y: 0,
        opacity: 1,
        duration: 0.8,
        stagger,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top 85%',
          toggleActions: 'play none none reverse',
        },
      }
    );

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => {
        if (trigger.trigger === containerRef.current) {
          trigger.kill();
        }
      });
    };
  }, [stagger]);

  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
}

// Text reveal with character animation (Organimo style)
interface TextRevealProps {
  children: string;
  className?: string;
  emphasize?: number[]; // Indices of letters to emphasize
}

export function TextReveal({ children, className = '', emphasize = [] }: TextRevealProps) {
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!textRef.current) return;

    const chars = textRef.current.querySelectorAll('.char');

    gsap.fromTo(
      chars,
      { y: 20, opacity: 0 },
      {
        y: 0,
        opacity: 1,
        duration: 0.5,
        stagger: 0.03,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: textRef.current,
          start: 'top 90%',
          toggleActions: 'play none none reverse',
        },
      }
    );
  }, []);

  return (
    <div ref={textRef} className={className}>
      {children.split('').map((char, i) => (
        <span
          key={i}
          className={`char inline-block ${emphasize.includes(i) ? 'text-blue-400 font-bold' : ''}`}
        >
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </div>
  );
}
