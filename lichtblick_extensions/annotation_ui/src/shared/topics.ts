export const TOPICS = {
  state: "/annotation/state",
  cmd: "/annotation/cmd",
  trajectoryDet: "/annotation/data/trajectory/deterministic",
  trajectoryStoch: "/annotation/data/trajectory/stochastic",
  trajectoryGt: "/annotation/data/trajectory/ground_truth",
  trajectoryEgoHistory: "/annotation/data/trajectory/ego_history",
  trajectoryGtSnippet: "/annotation/data/trajectory/gt_snippet",
  mapMarkers: "/annotation/data/map_markers",
  trackedObjects: "/annotation/data/tracked_objects",
  footprints: "/annotation/data/footprints",
} as const;
