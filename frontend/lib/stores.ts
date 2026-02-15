import { create } from 'zustand';

// =============================================================================
// Auth Store
// =============================================================================

interface AuthState {
  user: { id: string; username: string; email: string } | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string, user: { id: string; username: string; email: string }) => void;
  logout: () => void;
  loadFromStorage: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,

  login: (token, user) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));
    }
    set({ token, user, isAuthenticated: true });
  },

  logout: () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
    set({ token: null, user: null, isAuthenticated: false });
  },

  loadFromStorage: () => {
    if (typeof window === 'undefined') return;
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');
    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);
        set({ token, user, isAuthenticated: true });
      } catch {
        set({ token: null, user: null, isAuthenticated: false });
      }
    }
  },
}));


// =============================================================================
// Twin Store
// =============================================================================

interface TwinState {
  activeTwinId: string | null;
  twinStatus: string;
  twinData: Record<string, unknown> | null;
  simulationResults: Record<string, unknown> | null;
  qasmCircuits: Record<string, string>;
  setActiveTwin: (id: string | null) => void;
  setTwinStatus: (status: string) => void;
  setTwinData: (data: Record<string, unknown> | null) => void;
  setSimulationResults: (results: Record<string, unknown> | null) => void;
  setQasmCircuits: (circuits: Record<string, string>) => void;
  reset: () => void;
}

export const useTwinStore = create<TwinState>((set) => ({
  activeTwinId: null,
  twinStatus: 'draft',
  twinData: null,
  simulationResults: null,
  qasmCircuits: {},

  setActiveTwin: (id) => set({ activeTwinId: id }),
  setTwinStatus: (status) => set({ twinStatus: status }),
  setTwinData: (data) => set({ twinData: data }),
  setSimulationResults: (results) => set({ simulationResults: results }),
  setQasmCircuits: (circuits) => set({ qasmCircuits: circuits }),
  reset: () =>
    set({
      activeTwinId: null,
      twinStatus: 'draft',
      twinData: null,
      simulationResults: null,
      qasmCircuits: {},
    }),
}));


// =============================================================================
// Benchmark Store
// =============================================================================

interface BenchmarkState {
  activeModule: string | null;
  isRunning: boolean;
  results: Record<string, unknown> | null;
  qasmCircuit: string | null;
  setActiveModule: (module: string | null) => void;
  setIsRunning: (running: boolean) => void;
  setResults: (results: Record<string, unknown> | null) => void;
  setQasmCircuit: (qasm: string | null) => void;
}

export const useBenchmarkStore = create<BenchmarkState>((set) => ({
  activeModule: null,
  isRunning: false,
  results: null,
  qasmCircuit: null,

  setActiveModule: (module) => set({ activeModule: module }),
  setIsRunning: (running) => set({ isRunning: running }),
  setResults: (results) => set({ results }),
  setQasmCircuit: (qasm) => set({ qasmCircuit: qasm }),
}));
