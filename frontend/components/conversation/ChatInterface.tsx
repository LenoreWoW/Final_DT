'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Cpu, Loader2, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { twinService, Message } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ChatInterfaceProps {
  onTwinCreated: (twinId: string) => void;
  onStatusChange: (status: string) => void;
}

export function ChatInterface({ onTwinCreated, onStatusChange }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentTwinId, setCurrentTwinId] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initial greeting
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          role: 'assistant',
          content: "I'm the Universal Twin Builder. Describe any system — a marathon runner, a forest fire, a supply chain, or a molecule — and I'll generate a quantum digital twin for you.",
        },
      ]);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await twinService.sendMessage(input, currentTwinId);
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      
      if (response.twin_id && !currentTwinId) {
        setCurrentTwinId(response.twin_id);
        onTwinCreated(response.twin_id);
      }

      if (response.twin_status) {
        onStatusChange(response.twin_status);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error connecting to the quantum engine.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] border border-white/10 rounded-xl bg-white/5 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-white/5 backdrop-blur-sm flex items-center gap-2">
        <Cpu className="w-5 h-5 text-cyan-400" />
        <h2 className="font-semibold text-white">Twin Builder</h2>
        {currentTwinId && (
          <span className="text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded-full ml-auto">
            Twin Active
          </span>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#0a0a0a]/50">
        <AnimatePresence>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "flex w-full",
                msg.role === 'user' ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed",
                  msg.role === 'user'
                    ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-br-none shadow-lg shadow-blue-500/20"
                    : "bg-white/5 border border-white/10 text-white/90 rounded-bl-none backdrop-blur-sm"
                )}
              >
                {msg.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2 text-xs font-medium text-cyan-400/80">
                    <Sparkles className="w-3 h-3" />
                    Quantum AI
                  </div>
                )}
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-white/5 border border-white/10 rounded-2xl rounded-bl-none p-4 backdrop-blur-sm flex items-center gap-3">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span className="text-sm text-white/50">Processing with quantum engine...</span>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="p-4 bg-white/5 border-t border-white/10">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your system..."
            className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50 transition-all"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}

