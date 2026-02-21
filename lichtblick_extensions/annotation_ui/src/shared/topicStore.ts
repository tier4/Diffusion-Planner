import type { PanelExtensionContext } from "@lichtblick/suite";

import { TOPICS } from "./topics";
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
  trajectory_messages: {
    deterministic: null,
    stochastic: null,
    ground_truth: null,
    ego_history: null,
    gt_snippet: null,
  },
};


let state: AnnotationState = defaultState;
let contextRef: PanelExtensionContext | null = null;
const listeners = new Set<(nextState: AnnotationState) => void>();
const trajectoryFingerprints: Record<string, string | null> = {
  deterministic: null,
  stochastic: null,
  ground_truth: null,
  ego_history: null,
  gt_snippet: null,
};

function trajectoryFingerprint(msg: unknown): string {
  const points = (msg as { points?: Array<{ pose?: { position?: { x?: number; y?: number; z?: number } } }> }).points ?? [];
  const n = points.length;
  if (n === 0) {
    return "n:0";
  }
  const first = points[0]?.pose?.position;
  const last = points[n - 1]?.pose?.position;
  const mid = points[Math.floor(n / 2)]?.pose?.position;
  return [
    `n:${n}`,
    `f:${first?.x ?? 0},${first?.y ?? 0},${first?.z ?? 0}`,
    `m:${mid?.x ?? 0},${mid?.y ?? 0},${mid?.z ?? 0}`,
    `l:${last?.x ?? 0},${last?.y ?? 0},${last?.z ?? 0}`,
  ].join("|");
}

function updateTrajectoryIfChanged(
  key: keyof NonNullable<AnnotationState["trajectory_messages"]>,
  nextMsg: unknown,
): boolean {
  const nextFp = trajectoryFingerprint(nextMsg);
  if (trajectoryFingerprints[key] === nextFp) {
    return false;
  }
  trajectoryFingerprints[key] = nextFp;
  return true;
}

function notify() {
  listeners.forEach((listener) => listener(state));
}

export function getState(): AnnotationState {
  return state;
}

export function subscribe(listener: (nextState: AnnotationState) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function initTopicStore(context: PanelExtensionContext): void {
  contextRef = context;
  context.advertise?.(TOPICS.cmd, "std_msgs/msg/String");
  context.subscribe([
    { topic: TOPICS.state },
    { topic: TOPICS.trajectoryDet },
    { topic: TOPICS.trajectoryStoch },
    { topic: TOPICS.trajectoryGt },
    { topic: TOPICS.trajectoryEgoHistory },
    { topic: TOPICS.trajectoryGtSnippet },
  ]);
  context.watch("currentFrame");
  context.onRender = (renderState, done) => {
    try {
      if (renderState.currentFrame) {
        for (const msgEvent of renderState.currentFrame) {
           const tm = state.trajectory_messages ?? {
            deterministic: null,
            stochastic: null,
            ground_truth: null,
            ego_history: null,
            gt_snippet: null,
          };
          if (msgEvent.topic === TOPICS.state) {
            const messageText = (msgEvent.message as { data?: string }).data;
            if (messageText) {
              const payload = JSON.parse(messageText) as Partial<AnnotationState>;
              state = {
                ...state,
                ...payload,
                isLoading: false,
                lastUpdateNote: `Last update done at ${new Date().toLocaleTimeString()}`,
              };
            }
          } else if (msgEvent.topic === TOPICS.trajectoryDet) {
            if (!updateTrajectoryIfChanged("deterministic", msgEvent.message)) {
              continue;
            }
            state = {
              ...state,
              trajectory_messages: {
                ...tm,
                deterministic: msgEvent.message as NonNullable<AnnotationState["trajectory_messages"]>["deterministic"],
              },
            };
          } else if (msgEvent.topic === TOPICS.trajectoryStoch) {
            if (!updateTrajectoryIfChanged("stochastic", msgEvent.message)) {
              continue;
            }
            state = {
              ...state,
              trajectory_messages: {
                ...tm,
                stochastic: msgEvent.message as NonNullable<AnnotationState["trajectory_messages"]>["stochastic"],
              },
            };
          } else if (msgEvent.topic === TOPICS.trajectoryGt) {
            if (!updateTrajectoryIfChanged("ground_truth", msgEvent.message)) {
              continue;
            }
            state = {
              ...state,
              trajectory_messages: {
                ...tm,
                ground_truth: msgEvent.message as NonNullable<AnnotationState["trajectory_messages"]>["ground_truth"],
              },
            };
          } else if (msgEvent.topic === TOPICS.trajectoryEgoHistory) {
            if (!updateTrajectoryIfChanged("ego_history", msgEvent.message)) {
              continue;
            }
            state = {
              ...state,
              trajectory_messages: {
                ...tm,
                ego_history: msgEvent.message as NonNullable<AnnotationState["trajectory_messages"]>["ego_history"],
              },
            };
          } else if (msgEvent.topic === TOPICS.trajectoryGtSnippet) {
            if (!updateTrajectoryIfChanged("gt_snippet", msgEvent.message)) {
              continue;
            }
            state = {
              ...state,
              trajectory_messages: {
                ...tm,
                gt_snippet: msgEvent.message as NonNullable<AnnotationState["trajectory_messages"]>["gt_snippet"],
              },
            };
          }
        }
      }
      notify();
    } catch (error) {
      state = { ...state, lastError: String(error) };
      notify();
    } finally {
      done();
    }
  };
}

export function publishCommand(message: WsMessage): void {
  if (!contextRef) {
    state = { ...state, lastError: "Panel context not initialized" };
    notify();
    return;
  }
  const payload: WsMessage = {
    ...message,
    request_id: message.request_id ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  };
  state = {
    ...state,
    isLoading: true,
    loadingLabel: `Updating: ${message.type}`,
    lastUpdateNote: undefined,
  };
  notify();
  contextRef.publish?.(TOPICS.cmd, { data: JSON.stringify(payload) });
}