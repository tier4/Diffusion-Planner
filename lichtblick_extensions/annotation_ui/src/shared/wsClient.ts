import { DEFAULT_WS_URL } from "./config";
import type { AnnotationState, WsMessage } from "./types";

const defaultState: AnnotationState = {
  texts: {
    metric: "",
    progress: "",
    metrics: "",
    metrics_full_table: "",
    metrics_ade_fde_table: "",
    sidebar: "",
    history: "",
  },
  plots: {
    trajectory: null,
    velocity: null,
    lateral: null,
  },
  params: {
    noise_scale: 2.5,
    fde_threshold: 2.0,
    ade_threshold: 1.0,
    max_retries: 50,
    zoom_level: 5,
    time_step: 40,
    gt_similarity_mode: true,
  },
  status: {
    current_index: 0,
    total_samples: 0,
    total_preferences: 0,
    target_count: 0,
    annotation_complete: false,
    current_filter: "All",
    auto_skip_labeled: false,
    current_jump_size: 1,
  },
};

let socket: WebSocket | null = null;
let state: AnnotationState = defaultState;
const listeners = new Set<(nextState: AnnotationState) => void>();
let reconnectTimer: number | null = null;
const pendingMessages: WsMessage[] = [];
const loadingActions = new Set([
  "load_sample",
  "regenerate",
  "select_winner",
  "jump",
  "jump_to_index",
  "jump_to_next_unlabeled",
  "toggle_filter",
  "update_time",
  "update_zoom",
  "launch_training",
]);

function notify(nextState: AnnotationState) {
  listeners.forEach((listener) => listener(nextState));
}

export function getState(): AnnotationState {
  return state;
}

export function subscribe(listener: (nextState: AnnotationState) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function ensureConnected(url: string = DEFAULT_WS_URL) {
  if (socket && socket.readyState <= WebSocket.OPEN) {
    return;
  }

  socket = new WebSocket(url);
  state = { ...state, lastError: undefined };
  notify(state);

  socket.onopen = () => {
    sendMessage({ type: "hello" });
    sendMessage({ type: "get_state" });
    while (pendingMessages.length > 0 && socket && socket.readyState === WebSocket.OPEN) {
      const queued = pendingMessages.shift();
      if (queued) {
        socket.send(JSON.stringify(queued));
      }
    }
  };

  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data) as WsMessage;
      if (message.type === "state_update" && message.payload) {
        const payload = message.payload as Partial<AnnotationState>;
        state = {
          ...state,
          ...payload,
          isLoading: false,
          lastUpdateNote: `Last update done at ${new Date().toLocaleTimeString()}`,
        };
        notify(state);
      } else if (message.type === "hello_ack" || message.type === "pong") {
        return;
      } else if (message.type === "error" && message.payload) {
        state = { ...state, lastError: String(message.payload["message"] ?? "Unknown error") };
        notify(state);
      }
    } catch (err) {
      state = { ...state, lastError: String(err) };
      notify(state);
    }
  };

  socket.onclose = () => {
    socket = null;
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
    }
    reconnectTimer = window.setTimeout(() => {
      ensureConnected(url);
    }, 1500);
  };
}

export function sendMessage(message: WsMessage, url?: string) {
  const enriched: WsMessage = {
    ...message,
    request_id: message.request_id ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  };
  ensureConnected(url);
  if (loadingActions.has(message.type)) {
    state = {
      ...state,
      isLoading: true,
      loadingLabel: `Updating: ${message.type}`,
      lastUpdateNote: undefined,
    };
    notify(state);
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    pendingMessages.push(enriched);
    return;
  }
  socket.send(JSON.stringify(enriched));
}
