import React, { useEffect, useState } from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { ui } from "../shared/ui";

export function ControlsPanel({ context }: { context: PanelExtensionContext }) {
  const { params } = useWsState(context);
  const [localParams, setLocalParams] = useState(params);

  useEffect(() => {
    setLocalParams(params);
  }, [params]);

  const updateParam = (key: keyof typeof localParams, value: number | boolean) => {
    const next = { ...localParams, [key]: value };
    setLocalParams(next);
    sendMessage({ type: "set_params", payload: { [key]: value } });
  };

  return (
    <div style={ui.page}>
      <h3 style={ui.title}>⚙️ Annotation Controls</h3>
      <div style={{ ...ui.section, borderLeft: "4px solid #8b5cf6" }}>
        <h4 style={ui.subtitle}>🎛️ Generation Parameters</h4>
        <label style={{ display: "block", marginBottom: "12px", fontSize: "16px", fontWeight: 600 }}>
          Noise Scale: {localParams.noise_scale.toFixed(1)}
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={localParams.noise_scale}
            onChange={(event) => updateParam("noise_scale", Number(event.target.value))}
            style={ui.slider}
          />
        </label>

        <label style={{ display: "block", marginBottom: "12px", fontSize: "16px", fontWeight: 600 }}>
          FDE Threshold: {localParams.fde_threshold.toFixed(1)}
          <input
            type="range"
            min={0.5}
            max={10.0}
            step={0.1}
            value={localParams.fde_threshold}
            onChange={(event) => updateParam("fde_threshold", Number(event.target.value))}
            style={ui.slider}
          />
        </label>

        <label style={{ display: "block", marginBottom: "12px", fontSize: "16px", fontWeight: 600 }}>
          ADE Threshold: {localParams.ade_threshold.toFixed(1)}
          <input
            type="range"
            min={0.1}
            max={5.0}
            step={0.1}
            value={localParams.ade_threshold}
            onChange={(event) => updateParam("ade_threshold", Number(event.target.value))}
            style={ui.slider}
          />
        </label>

        <label style={{ display: "block", marginBottom: "12px", fontSize: "16px", fontWeight: 600 }}>
          Max Retries: {localParams.max_retries}
          <input
            type="range"
            min={10}
            max={200}
            step={10}
            value={localParams.max_retries}
            onChange={(event) => updateParam("max_retries", Number(event.target.value))}
            style={ui.slider}
          />
        </label>

        <label style={{ display: "block", marginBottom: "8px", fontSize: "15px" }}>
          <input
            type="checkbox"
            checked={localParams.gt_similarity_mode}
            onChange={(event) => updateParam("gt_similarity_mode", event.target.checked)}
          />
          <span style={{ marginLeft: "6px" }}>🎯 GT Similarity Mode</span>
        </label>
      </div>

      <div style={{ ...ui.section, borderLeft: "4px solid #0ea5e9" }}>
        <h4 style={ui.subtitle}>🖼️ Visualization</h4>
        <label style={{ display: "block", marginBottom: "12px", fontSize: "16px", fontWeight: 600 }}>
          Zoom Level: {localParams.zoom_level}
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={localParams.zoom_level}
            onChange={(event) => {
              const value = Number(event.target.value);
              updateParam("zoom_level", value);
              sendMessage({ type: "update_zoom", payload: { zoom_level: value } });
            }}
            style={ui.slider}
          />
        </label>

        <label style={{ display: "block", marginBottom: "8px", fontSize: "16px", fontWeight: 600 }}>
          Time Step: {localParams.time_step}
          <input
            type="range"
            min={0}
            max={79}
            step={1}
            value={localParams.time_step}
            onChange={(event) => {
              const value = Number(event.target.value);
              updateParam("time_step", value);
              sendMessage({ type: "update_time", payload: { time_step: value } });
            }}
            style={ui.slider}
          />
        </label>
      </div>
    </div>
  );
}
