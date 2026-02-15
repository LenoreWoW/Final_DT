/**
 * WebSocket client for real-time twin generation and simulation progress.
 *
 * Connects to ws://localhost:8000/ws/{twinId} and dispatches typed messages.
 */

export type WSMessageType =
  | 'connected'
  | 'generation_progress'
  | 'simulation_progress'
  | 'simulation_complete'
  | 'error'
  | 'pong'
  | 'ack';

export interface WSMessage {
  type: WSMessageType;
  twin_id?: string;
  step?: string | number;
  progress?: number;
  total?: number;
  results?: Record<string, unknown>;
  detail?: string;
  timestamp?: string;
}

export type WSEventHandler = (message: WSMessage) => void;

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

export class TwinWebSocket {
  private ws: WebSocket | null = null;
  private twinId: string;
  private handlers: Map<WSMessageType | '*', WSEventHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private shouldReconnect = true;

  constructor(twinId: string) {
    this.twinId = twinId;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(`${WS_BASE}/${this.twinId}`);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          this._dispatch(message);
        } catch {
          // ignore malformed messages
        }
      };

      this.ws.onclose = () => {
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
        }
      };

      this.ws.onerror = () => {
        this._dispatch({ type: 'error', detail: 'WebSocket connection error' });
      };
    } catch {
      // WebSocket constructor can throw if URL is invalid
    }
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(action: string, data?: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action, ...data }));
    }
  }

  ping(): void {
    this.send('ping');
  }

  on(type: WSMessageType | '*', handler: WSEventHandler): () => void {
    const existing = this.handlers.get(type) || [];
    existing.push(handler);
    this.handlers.set(type, existing);

    // Return unsubscribe function
    return () => {
      const list = this.handlers.get(type) || [];
      this.handlers.set(
        type,
        list.filter((h) => h !== handler)
      );
    };
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private _dispatch(message: WSMessage): void {
    // Fire type-specific handlers
    const typeHandlers = this.handlers.get(message.type) || [];
    typeHandlers.forEach((h) => h(message));

    // Fire wildcard handlers
    const wildcardHandlers = this.handlers.get('*') || [];
    wildcardHandlers.forEach((h) => h(message));
  }
}

/**
 * Create and auto-connect a WebSocket for a twin.
 * Returns a cleanup function for useEffect.
 */
export function createTwinSocket(
  twinId: string,
  onMessage: WSEventHandler
): { socket: TwinWebSocket; cleanup: () => void } {
  const socket = new TwinWebSocket(twinId);
  const unsub = socket.on('*', onMessage);
  socket.connect();

  return {
    socket,
    cleanup: () => {
      unsub();
      socket.disconnect();
    },
  };
}
