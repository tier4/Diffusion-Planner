import type { Trajectory } from "autoware-lichtblick-plugins/src/msgs/planning/Trajectory";

export interface AnnotationTexts {
  metric: string;
  progress: string;
  metrics: string;
  metrics_full_table: string;
  metrics_ade_fde_table: string;
  sidebar: string;
  history: string;
}

export interface AnnotationPlots {
  trajectory: string | null;
  velocity: string | null;
  lateral: string | null;
}

export interface AnnotationParams {
  noise_scale: number;
  fde_threshold: number;
  ade_threshold: number;
  max_retries: number;
  zoom_level: number;
  time_step: number;
  gt_similarity_mode: boolean;
}

export interface AnnotationStatus {
  current_index: number;
  total_samples: number;
  total_preferences: number;
  target_count: number;
  annotation_complete: boolean;
  current_filter: string;
  auto_skip_labeled: boolean;
  current_jump_size: number;
}

export interface AnnotationState {
  texts: AnnotationTexts;
  plots: AnnotationPlots;
  params: AnnotationParams;
  status: AnnotationStatus;
  training?: {
    phase: string;
    message: string;
    epoch: number;
    total_epochs: number;
    batch: number;
    total_batches: number;
    metrics?: {
      loss?: number;
      accuracy?: number;
      reward_margin?: number;
    };
  };
  trajectory_messages?: {
    deterministic: Trajectory | null;
    stochastic: Trajectory | null;
    ground_truth: Trajectory | null;
    ego_history?: Trajectory | null;
    gt_snippet?: Trajectory | null;
  };
  isLoading?: boolean;
  loadingLabel?: string;
  lastUpdateNote?: string;
  lastError?: string;
}

export interface WsMessage {
  type: string;
  payload?: Record<string, unknown>;
  request_id?: string;
}
