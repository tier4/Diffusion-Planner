import React, { useMemo, useState } from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { computeAccelerationMps2, computeVelocitiesKmH, toSeriesPoints } from "../shared/messageAdapters";

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
  return values[idx]!.toFixed(2);
}

export function VelocityPlotPanel({ context }: { context: PanelExtensionContext }) {
  const { trajectory_messages, isLoading, params } = useWsState(context);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const detVel = computeVelocitiesKmH(trajectory_messages?.deterministic ?? null);
  const stochVel = computeVelocitiesKmH(trajectory_messages?.stochastic ?? null);
  const gtVel = computeVelocitiesKmH(trajectory_messages?.ground_truth ?? null);
  const detAcc = computeAccelerationMps2(detVel);
  const stochAcc = computeAccelerationMps2(stochVel);

  const width = 720;
  const subHeight = 220;
  const xMaxVel = Math.max(detVel.length, stochVel.length, gtVel.length) - 1;
  const xMaxAcc = Math.max(detAcc.length, stochAcc.length) - 1;
  const velScale = useMemo(() => createScales(width, subHeight, xMaxVel, 0, 60), [width, subHeight, xMaxVel]);
  const accScale = useMemo(() => createScales(width, subHeight, xMaxAcc, -2.5, 2.5), [width, subHeight, xMaxAcc]);
  const velDetPath = linePath(toSeriesPoints(detVel), velScale.xToPx, velScale.yToPx);
  const velStochPath = linePath(toSeriesPoints(stochVel), velScale.xToPx, velScale.yToPx);
  const velGtPath = linePath(toSeriesPoints(gtVel), velScale.xToPx, velScale.yToPx);
  const accDetPath = linePath(toSeriesPoints(detAcc), accScale.xToPx, accScale.yToPx);
  const accStochPath = linePath(toSeriesPoints(stochAcc), accScale.xToPx, accScale.yToPx);

  const hasData = detVel.length > 0 || stochVel.length > 0;
  const activeIdx = hoverIdx ?? params.time_step;
  const activeVelX = velScale.xToPx(clamp(activeIdx, 0, Math.max(xMaxVel, 1)));
  const activeAccX = accScale.xToPx(clamp(activeIdx, 0, Math.max(xMaxAcc, 1)));

  const yTicksVel = [0, 15, 30, 45, 60];
  const yTicksAcc = [-2.5, -1.25, 0, 1.25, 2.5];

  const onMouseMove = (evt: React.MouseEvent<SVGRectElement>, scale: ReturnType<typeof createScales>, maxIdx: number) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const idx = Math.round(scale.pxToX((x / rect.width) * width));
    setHoverIdx(clamp(idx, 0, Math.max(maxIdx, 0)));
  };

  return (
    <div style={{ padding: "14px", fontFamily: "\"Inter\", sans-serif" }}>
      <h3>Velocity Plot</h3>
      <div style={{ fontSize: "13px", color: "#4b5563", marginBottom: "8px" }}>
        Hover to inspect values, click chart to set time step, current step: {activeIdx}
      </div>
      {isLoading ? (
        <div style={{ color: "#9a3412" }}>Updating trajectory series...</div>
      ) : hasData ? (
        <svg viewBox={`0 0 ${width} ${subHeight * 2 + 48}`} style={{ width: "100%", border: "1px solid #d1d5db", borderRadius: "10px", background: "#fff" }}>
          <text x="16" y="16" fontSize="13" fontWeight="700">Velocity (km/h)</text>
          {yTicksVel.map((y) => (
            <g key={`vy-${y}`}>
              <line x1={velScale.frame.left} y1={velScale.yToPx(y)} x2={width - velScale.frame.right} y2={velScale.yToPx(y)} stroke="#eef2f7" />
              <text x={12} y={velScale.yToPx(y) + 4} fontSize="11" fill="#6b7280">{y}</text>
            </g>
          ))}
          <line x1={velScale.frame.left} y1={velScale.frame.top} x2={velScale.frame.left} y2={subHeight - velScale.frame.bottom} stroke="#9ca3af" />
          <line x1={velScale.frame.left} y1={subHeight - velScale.frame.bottom} x2={width - velScale.frame.right} y2={subHeight - velScale.frame.bottom} stroke="#9ca3af" />
          <path d={velDetPath} stroke="#16a34a" strokeWidth="2.2" fill="none" />
          <path d={velStochPath} stroke="#ea580c" strokeWidth="2.2" fill="none" />
          <path d={velGtPath} stroke="#dc2626" strokeWidth="2" fill="none" strokeDasharray="5 4" />
          <line x1={activeVelX} y1={velScale.frame.top} x2={activeVelX} y2={subHeight - velScale.frame.bottom} stroke="#2563eb" strokeDasharray="4 4" />
          <rect
            x={velScale.frame.left}
            y={velScale.frame.top}
            width={velScale.frame.innerW}
            height={velScale.frame.innerH}
            fill="transparent"
            style={{ cursor: "crosshair" }}
            onMouseMove={(evt) => onMouseMove(evt, velScale, xMaxVel)}
            onMouseLeave={() => setHoverIdx(null)}
            onClick={() => sendMessage({ type: "update_time", payload: { time_step: activeIdx } })}
          />

          <g transform={`translate(0, ${subHeight + 28})`}>
            <text x="16" y="16" fontSize="13" fontWeight="700">Longitudinal acceleration (m/s²)</text>
            {yTicksAcc.map((y) => (
              <g key={`ay-${y}`}>
                <line x1={accScale.frame.left} y1={accScale.yToPx(y)} x2={width - accScale.frame.right} y2={accScale.yToPx(y)} stroke="#eef2f7" />
                <text x={8} y={accScale.yToPx(y) + 4} fontSize="11" fill="#6b7280">{y}</text>
              </g>
            ))}
            <line x1={accScale.frame.left} y1={accScale.frame.top} x2={accScale.frame.left} y2={subHeight - accScale.frame.bottom} stroke="#9ca3af" />
            <line x1={accScale.frame.left} y1={subHeight - accScale.frame.bottom} x2={width - accScale.frame.right} y2={subHeight - accScale.frame.bottom} stroke="#9ca3af" />
            <path d={accDetPath} stroke="#16a34a" strokeWidth="2.2" fill="none" />
            <path d={accStochPath} stroke="#ea580c" strokeWidth="2.2" fill="none" />
            <line x1={activeAccX} y1={accScale.frame.top} x2={activeAccX} y2={subHeight - accScale.frame.bottom} stroke="#2563eb" strokeDasharray="4 4" />
            <rect
              x={accScale.frame.left}
              y={accScale.frame.top}
              width={accScale.frame.innerW}
              height={accScale.frame.innerH}
              fill="transparent"
              style={{ cursor: "crosshair" }}
              onMouseMove={(evt) => onMouseMove(evt, accScale, xMaxAcc)}
              onMouseLeave={() => setHoverIdx(null)}
              onClick={() => sendMessage({ type: "update_time", payload: { time_step: activeIdx } })}
            />
          </g>

          <g transform={`translate(${width - 250}, 18)`}>
            <rect x={0} y={0} width={240} height={92} rx={8} fill="#f8fafc" stroke="#e2e8f0" />
            <text x={10} y={18} fontSize="12" fontWeight="700">Time step {activeIdx} ({(activeIdx * 0.1).toFixed(1)}s)</text>
            <text x={10} y={36} fontSize="11" fill="#166534">Det vel: {nearestValue(detVel, activeIdx)} km/h</text>
            <text x={10} y={52} fontSize="11" fill="#c2410c">Stoch vel: {nearestValue(stochVel, activeIdx)} km/h</text>
            <text x={10} y={68} fontSize="11" fill="#dc2626">GT vel: {nearestValue(gtVel, activeIdx)} km/h</text>
            <text x={10} y={84} fontSize="11" fill="#1f2937">
              Det/Stoch accel: {nearestValue(detAcc, activeIdx)} / {nearestValue(stochAcc, activeIdx)} m/s²
            </text>
          </g>
        </svg>
      ) : (
        <div>No trajectory message data available</div>
      )}
    </div>
  );
}
