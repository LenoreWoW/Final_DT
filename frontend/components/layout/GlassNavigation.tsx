'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Atom } from 'lucide-react';

export default function GlassNavigation() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled
            ? 'bg-black/40 backdrop-blur-xl border-b border-white/10 shadow-lg shadow-purple-500/10'
            : 'bg-transparent'
        }`}
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link
              href="/"
              className="flex items-center gap-3 group"
            >
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
                className="p-2 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg"
              >
                <Atom className="w-6 h-6 text-white" />
              </motion.div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Quantum Twin
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-8">
              <NavLink href="/builder">Builder</NavLink>
              <NavLink href="/showcase">Showcase</NavLink>
              <NavLink href="/dashboard">Dashboard</NavLink>

              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  href="/builder"
                  className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium text-white hover:shadow-lg hover:shadow-purple-500/50 transition-shadow"
                >
                  Start Building
                </Link>
              </motion.div>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-white hover:bg-white/10 rounded-lg transition"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={{ type: 'spring', damping: 25 }}
            className="fixed inset-0 z-40 bg-black/95 backdrop-blur-xl md:hidden"
          >
            <div className="flex flex-col items-center justify-center h-full gap-8 text-2xl">
              <MobileNavLink href="/builder" onClick={() => setMobileMenuOpen(false)}>
                Builder
              </MobileNavLink>
              <MobileNavLink href="/showcase" onClick={() => setMobileMenuOpen(false)}>
                Showcase
              </MobileNavLink>
              <MobileNavLink href="/dashboard" onClick={() => setMobileMenuOpen(false)}>
                Dashboard
              </MobileNavLink>

              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  href="/builder"
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium text-white"
                >
                  Start Building
                </Link>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link href={href} className="relative group text-white/70 hover:text-white transition">
      <span>{children}</span>
      <motion.div
        className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-400 to-purple-400"
        initial={{ scaleX: 0 }}
        whileHover={{ scaleX: 1 }}
        transition={{ duration: 0.3 }}
      />
    </Link>
  );
}

function MobileNavLink({
  href,
  children,
  onClick,
}: {
  href: string;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}>
      <Link
        href={href}
        onClick={onClick}
        className="text-white/70 hover:text-white transition"
      >
        {children}
      </Link>
    </motion.div>
  );
}
