export const TrajectorySettings = {};

export function convertTrajectory(): any {
  // Panels consume trajectory topics directly; this converter enables 3D panel compatibility.
  return { entities: [], deletions: [] };
}

