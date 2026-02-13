import React, { useMemo, useState } from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import {
  computeCurvature,
  computeLateralAccelerationMps2,
  computeVelocitiesKmH,
  toSeriesPoints,
} from "../shared/messageAdapters";

type Pt = { x: number; y: number };

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function createScales(width: number, height: number, xMax: number, yMin: number, yMax: number) {
  const left = 52;
  const right = 16;
  const top = 34;
  const bottom = 28;
  const innerW = Math.max(width - left - right, 10);
  const innerH = Math.max(height - top - bottom, 10);
  const spanY = Math.max(yMax - yMin, 1e-6);
  return {
    frame: { left, right, top, bottom, innerW, innerH },
    xToPx: (x: number) => left + (x / Math.max(xMax, 1)) * innerW,
    yToPx: (y: number) => top + (1 - (y - yMin) / spanY) * innerH,
    pxToX: (px: number) => ((px - left) / innerW) * Math.max(xMax, 1),
  };
}

function linePath(points: Pt[], xToPx: (x: number) => number, yToPx: (y: number) => number): string {
  if (points.length === 0) {
    return "";
  }
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x)} ${yToPx(p.y)}`).join(" ");
}

function nearestValue(values: number[], idx: number): string {
  if (idx < 0 || idx >= values.length) {
    return "-";
  }
  return values[idx]!.toFixed(3);
}

export function LateralPlotPanel({ context }: { context: PanelExtensionContext }) {
  const { trajectory_messages, isLoading, params } = useWsState(context);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const detCurv = computeCurvature(trajectory_messages?.deterministic ?? null);
  const stochCurv = computeCurvature(trajectory_messages?.stochastic ?? null);
  const gtCurv = computeCurvature(trajectory_messages?.ground_truth ?? null);
  const detVel = computeVelocitiesKmH(trajectory_messages?.deterministic ?? null);
  const stochVel = computeVelocitiesKmH(trajectory_messages?.stochastic ?? null);
  const gtVel = computeVelocitiesKmH(trajectory_messages?.ground_truth ?? null);
  const detLat = computeLateralAccelerationMps2(detCurv, detVel);
  const stochLat = computeLateralAccelerationMps2(stochCurv, stochVel);
  const gtLat = computeLateralAccelerationMps2(gtCurv, gtVel);

  const width = 720;
  const subHeight = 220;
  const xMaxLat = Math.max(detLat.length, stochLat.length, gtLat.length) - 1;
  const xMaxCurv = Math.max(detCurv.length, stochCurv.length, gtCurv.length) - 1;
  const latScale = useMemo(() => createScales(width, subHeight, xMaxLat, 0, 8), [width, subHeight, xMaxLat]);
  const curvScale = useMemo(() => createScales(width, subHeight, xMaxCurv, -0.2, 0.2), [width, subHeight, xMaxCurv]);
  const latDetPath = linePath(toSeriesPoints(detLat), latScale.xToPx, latScale.yToPx);
  const latStochPath = linePath(toSeriesPoints(stochLat), latScale.xToPx, latScale.yToPx);
  const latGtPath = linePath(toSeriesPoints(gtLat), latScale.xToPx, latScale.yToPx);
  const curvDetPath = linePath(toSeriesPoints(detCurv), curvScale.xToPx, curvScale.yToPx);
  const curvStochPath = linePath(toSeriesPoints(stochCurv), curvScale.xToPx, curvScale.yToPx);
  const curvGtPath = linePath(toSeriesPoints(gtCurv), curvScale.xToPx, curvScale.yToPx);

  const hasData = detCurv.length > 0 || stochCurv.length > 0;
  const activeIdx = hoverIdx ?? params.time_step;
  const activeLatX = latScale.xToPx(clamp(activeIdx, 0, Math.max(xMaxLat, 1)));
  const activeCurvX = curvScale.xToPx(clamp(activeIdx, 0, Math.max(xMaxCurv, 1)));
  const yTicksLat = [0, 2, 4, 6, 8];
  const yTicksCurv = [-0.2, -0.1, 0, 0.1, 0.2];

  const onMouseMove = (evt: React.MouseEvent<SVGRectElement>, scale: ReturnType<typeof createScales>, maxIdx: number) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const idx = Math.round(scale.pxToX((x / rect.width) * width));
    setHoverIdx(clamp(idx, 0, Math.max(maxIdx, 0)));
  };

  return (
    <div style={{ padding: "14px", fontFamily: "\"Inter\", sans-serif" }}>
      <h3>Lateral/Curvature Plot</h3>
      <div style={{ fontSize: "13px", color: "#4b5563", marginBottom: "8px" }}>
        Hover to inspect values, click chart to set time step, current step: {activeIdx}
      </div>
      {isLoading ? (
        <div style={{ color: "#9a3412" }}>Updating trajectory series...</div>
      ) : hasData ? (
        <svg viewBox={`0 0 ${width} ${subHeight * 2 + 48}`} style={{ width: "100%", border: "1px solid #d1d5db", borderRadius: "10px", background: "#fff" }}>
          <text x="16" y="16" fontSize="13" fontWeight="700">Lateral acceleration (m/s²)</text>
          {yTicksLat.map((y) => (
            <g key={`ly-${y}`}>
              <line x1={latScale.frame.left} y1={latScale.yToPx(y)} x2={width - latScale.frame.right} y2={latScale.yToPx(y)} stroke="#eef2f7" />
              <text x={12} y={latScale.yToPx(y) + 4} fontSize="11" fill="#6b7280">{y}</text>
            </g>
          ))}
          <line x1={latScale.frame.left} y1={latScale.frame.top} x2={latScale.frame.left} y2={subHeight - latScale.frame.bottom} stroke="#9ca3af" />
          <line x1={latScale.frame.left} y1={subHeight - latScale.frame.bottom} x2={width - latScale.frame.right} y2={subHeight - latScale.frame.bottom} stroke="#9ca3af" />
          <path d={latDetPath} stroke="#16a34a" strokeWidth="2.2" fill="none" />
          <path d={latStochPath} stroke="#ea580c" strokeWidth="2.2" fill="none" />
          <path d={latGtPath} stroke="#dc2626" strokeWidth="2" fill="none" strokeDasharray="5 4" />
          <line x1={activeLatX} y1={latScale.frame.top} x2={activeLatX} y2={subHeight - latScale.frame.bottom} stroke="#2563eb" strokeDasharray="4 4" />
          <rect
            x={latScale.frame.left}
            y={latScale.frame.top}
            width={latScale.frame.innerW}
            height={latScale.frame.innerH}
            fill="transparent"
            style={{ cursor: "crosshair" }}
            onMouseMove={(evt) => onMouseMove(evt, latScale, xMaxLat)}
            onMouseLeave={() => setHoverIdx(null)}
            onClick={() => sendMessage({ type: "update_time", payload: { time_step: activeIdx } })}
          />

          <g transform={`translate(0, ${subHeight + 28})`}>
            <text x="16" y="16" fontSize="13" fontWeight="700">Curvature (1/m)</text>
            {yTicksCurv.map((y) => (
              <g key={`cy-${y}`}>
                <line x1={curvScale.frame.left} y1={curvScale.yToPx(y)} x2={width - curvScale.frame.right} y2={curvScale.yToPx(y)} stroke="#eef2f7" />
                <text x={8} y={curvScale.yToPx(y) + 4} fontSize="11" fill="#6b7280">{y.toFixed(2)}</text>
              </g>
            ))}
            <line x1={curvScale.frame.left} y1={curvScale.frame.top} x2={curvScale.frame.left} y2={subHeight - curvScale.frame.bottom} stroke="#9ca3af" />
            <line x1={curvScale.frame.left} y1={subHeight - curvScale.frame.bottom} x2={width - curvScale.frame.right} y2={subHeight - curvScale.frame.bottom} stroke="#9ca3af" />
            <path d={curvDetPath} stroke="#16a34a" strokeWidth="2.2" fill="none" />
            <path d={curvStochPath} stroke="#ea580c" strokeWidth="2.2" fill="none" />
            <path d={curvGtPath} stroke="#dc2626" strokeWidth="2" fill="none" strokeDasharray="5 4" />
            <line x1={activeCurvX} y1={curvScale.frame.top} x2={activeCurvX} y2={subHeight - curvScale.frame.bottom} stroke="#2563eb" strokeDasharray="4 4" />
            <rect
              x={curvScale.frame.left}
              y={curvScale.frame.top}
              width={curvScale.frame.innerW}
              height={curvScale.frame.innerH}
              fill="transparent"
              style={{ cursor: "crosshair" }}
              onMouseMove={(evt) => onMouseMove(evt, curvScale, xMaxCurv)}
              onMouseLeave={() => setHoverIdx(null)}
              onClick={() => sendMessage({ type: "update_time", payload: { time_step: activeIdx } })}
            />
          </g>

          <g transform={`translate(${width - 250}, 18)`}>
            <rect x={0} y={0} width={240} height={92} rx={8} fill="#f8fafc" stroke="#e2e8f0" />
            <text x={10} y={18} fontSize="12" fontWeight="700">Time step {activeIdx} ({(activeIdx * 0.1).toFixed(1)}s)</text>
            <text x={10} y={36} fontSize="11" fill="#166534">Det lat: {nearestValue(detLat, activeIdx)} m/s²</text>
            <text x={10} y={52} fontSize="11" fill="#c2410c">Stoch lat: {nearestValue(stochLat, activeIdx)} m/s²</text>
            <text x={10} y={68} fontSize="11" fill="#dc2626">GT lat: {nearestValue(gtLat, activeIdx)} m/s²</text>
            <text x={10} y={84} fontSize="11" fill="#1f2937">
              Det/Stoch curv: {nearestValue(detCurv, activeIdx)} / {nearestValue(stochCurv, activeIdx)}
            </text>
          </g>
        </svg>
      ) : (
        <div>No trajectory message data available</div>
      )}
    </div>
  );
}
