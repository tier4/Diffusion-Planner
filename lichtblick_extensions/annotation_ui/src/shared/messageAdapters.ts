import type { Trajectory } from "autoware-lichtblick-plugins/src/msgs/planning/Trajectory";

const DT_SEC = 0.1;

type XY = { x: number; y: number };

function pointsFromTrajectory(traj: Trajectory | null): XY[] {
  if (!traj) {
    return [];
  }
  return traj.points.map((p) => ({ x: p.pose.position.x, y: p.pose.position.y }));
}

function orientationToHeading(orientation: { z: number; w: number }): number {
  return Math.atan2(2 * orientation.w * orientation.z, 1 - 2 * orientation.z * orientation.z);
}

export function computeVelocitiesKmH(traj: Trajectory | null): number[] {
  const pts = pointsFromTrajectory(traj);
  if (pts.length < 2) {
    return [];
  }
  const out: number[] = [];
  for (let i = 0; i < pts.length - 1; i += 1) {
    const dx = pts[i + 1]!.x - pts[i]!.x;
    const dy = pts[i + 1]!.y - pts[i]!.y;
    const mps = Math.hypot(dx, dy) / DT_SEC;
    out.push(mps * 3.6);
  }
  return out;
}

export function computeAccelerationMps2(velKmh: number[]): number[] {
  if (velKmh.length === 0) {
    return [];
  }
  const velMps = velKmh.map((v) => v / 3.6);
  const acc: number[] = [];
  for (let i = 0; i < velMps.length - 1; i += 1) {
    acc.push((velMps[i + 1]! - velMps[i]!) / DT_SEC);
  }
  acc.push(0);
  return acc;
}

export function computeCurvature(traj: Trajectory | null): number[] {
  if (!traj || traj.points.length < 2) {
    return [];
  }
  const headings = traj.points.map((p) => orientationToHeading(p.pose.orientation));
  const pts = pointsFromTrajectory(traj);
  const curvatures: number[] = [];
  for (let i = 0; i < headings.length - 1; i += 1) {
    const dHeading = headings[i + 1]! - headings[i]!;
    const dx = pts[i + 1]!.x - pts[i]!.x;
    const dy = pts[i + 1]!.y - pts[i]!.y;
    const ds = Math.hypot(dx, dy);
    curvatures.push(ds > 1e-6 ? dHeading / ds : 0);
  }
  return curvatures;
}

export function computeLateralAccelerationMps2(curvature: number[], velKmh: number[]): number[] {
  const len = Math.min(curvature.length, velKmh.length);
  const out: number[] = [];
  for (let i = 0; i < len; i += 1) {
    const mps = velKmh[i]! / 3.6;
    out.push(mps * mps * Math.abs(curvature[i]!));
  }
  return out;
}

export function toSeriesPoints(values: number[]): { x: number; y: number }[] {
  return values.map((y, idx) => ({ x: idx, y }));
}
