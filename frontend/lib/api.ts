import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// -------------------------------------------------------------------
// JWT Token Interceptor — attaches Authorization header from localStorage
// -------------------------------------------------------------------
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error: AxiosError) => Promise.reject(error)
);

// -------------------------------------------------------------------
// Error Interceptor — handles 401 and network errors globally
// -------------------------------------------------------------------
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      const status = error.response.status;

      // Unauthorized — clear token and redirect to login
      if (status === 401 && typeof window !== 'undefined') {
        localStorage.removeItem('token');
        // Only redirect if not already on login/register page
        const path = window.location.pathname;
        if (path !== '/login' && path !== '/register') {
          window.location.href = '/login';
        }
      }
    }

    return Promise.reject(error);
  }
);

// -------------------------------------------------------------------
// Shared Types
// -------------------------------------------------------------------
export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface Twin {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'generating' | 'active' | 'paused' | 'failed';
  domain?: string;
  extracted_system?: Record<string, unknown>;
  state?: Record<string, unknown>;
  quantum_metrics?: {
    qubits_allocated?: number;
    circuit_depth?: number;
    entanglement_pairs?: number;
    primary_algorithm?: string;
  };
  created_at: string;
}

export interface SimulationResult {
  twin_id: string;
  results: {
    scenarios?: Array<Record<string, unknown>>;
    statistics?: {
      scenarios_run?: number;
      mean_outcome?: number;
      std_deviation?: number;
    };
  };
  predictions: Array<Record<string, unknown>>;
  quantum_advantage: {
    speedup: number;
    classical_equivalent_seconds: number;
  };
  execution_time_seconds: number;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
}

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  created_at: string;
}

export interface BenchmarkModule {
  id: string;
  name: string;
  description: string;
  quantum_method: string;
  classical_method: string;
}

export interface BenchmarkResult {
  module: string;
  classical_time_seconds: number;
  quantum_time_seconds: number;
  classical_accuracy: number;
  quantum_accuracy: number;
  speedup: number;
  improvement: number;
  details: Record<string, unknown>;
}

// -------------------------------------------------------------------
// Auth Service
// -------------------------------------------------------------------
export const authService = {
  login: async (username: string, password: string): Promise<AuthResponse> => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },

  register: async (username: string, email: string, password: string): Promise<AuthResponse> => {
    const response = await api.post('/auth/register', { username, email, password });
    return response.data;
  },

  getMe: async (): Promise<UserProfile> => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// -------------------------------------------------------------------
// Twin Service (original + expanded)
// -------------------------------------------------------------------
export const twinService = {
  // Conversation
  sendMessage: async (message: string, twinId?: string) => {
    const response = await api.post('/conversation/', {
      message,
      twin_id: twinId,
    });
    return response.data;
  },

  getHistory: async (twinId: string) => {
    const response = await api.get(`/conversation/${twinId}/history`);
    return response.data;
  },

  // Twins
  getTwin: async (id: string): Promise<Twin> => {
    const response = await api.get(`/twins/${id}`);
    return response.data;
  },

  listTwins: async (): Promise<Twin[]> => {
    const response = await api.get('/twins/');
    return response.data;
  },

  // Simulation
  runSimulation: async (id: string, timeSteps: number = 100): Promise<SimulationResult> => {
    const response = await api.post(`/twins/${id}/simulate`, {
      time_steps: timeSteps,
      scenarios: 100,
    });
    return response.data;
  },

  // Query
  queryTwin: async (id: string, query: string) => {
    const response = await api.post(`/twins/${id}/query`, {
      query,
    });
    return response.data;
  },

  // QASM circuits
  getQASM: async (id: string): Promise<{ twin_id: string; twin_name: string; domain: string; circuits: Record<string, string>; circuit_count: number }> => {
    const response = await api.get(`/twins/${id}/qasm`);
    return response.data;
  },
};

// -------------------------------------------------------------------
// Benchmark Service
// -------------------------------------------------------------------
export const benchmarkService = {
  getModules: async (): Promise<BenchmarkModule[]> => {
    const response = await api.get('/benchmark/modules');
    return response.data.modules ?? response.data;
  },

  getResults: async (): Promise<BenchmarkResult[]> => {
    const response = await api.get('/benchmark/results');
    return response.data;
  },

  getModuleResults: async (moduleId: string): Promise<BenchmarkResult> => {
    const response = await api.get(`/benchmark/results/${moduleId}`);
    return response.data;
  },

  runBenchmark: async (
    moduleId: string,
    options?: { run_classical?: boolean; run_quantum?: boolean; parameters?: Record<string, unknown> }
  ): Promise<{ comparison: BenchmarkResult; raw_results: Record<string, unknown> }> => {
    const response = await api.post(`/benchmark/run/${moduleId}`, {
      run_classical: options?.run_classical ?? true,
      run_quantum: options?.run_quantum ?? true,
      parameters: options?.parameters ?? {},
    });
    return response.data;
  },
};

// -------------------------------------------------------------------
// Data Service
// -------------------------------------------------------------------
export const dataService = {
  uploadFile: async (file: File, twinId?: string): Promise<{ data: Record<string, unknown>; metadata: Record<string, unknown> }> => {
    const formData = new FormData();
    formData.append('file', file);
    if (twinId) {
      formData.append('twin_id', twinId);
    }

    const response = await api.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};
