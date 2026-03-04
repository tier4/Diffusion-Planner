import React from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { ui } from "../shared/ui";

export function SelectionPanel({ context }: { context: PanelExtensionContext }) {
  const { params, status } = useWsState(context);
  const greenOrangeDisabled = status.is_pruned && params.enable_initial_pruning;

  return (
    <div style={ui.page}>
      <h3 style={ui.title}>✅ Annotation Selection</h3>
      <div style={ui.section}>
        <button
          onClick={() => sendMessage({ type: "select_winner", payload: { winner: "trajectory_1" } })}
          disabled={greenOrangeDisabled}
          style={{
            width: "100%",
            marginBottom: "10px",
            borderRadius: "10px",
            padding: "12px",
            border: "none",
            color: "white",
            background: greenOrangeDisabled ? "#9ca3af" : "linear-gradient(90deg, #22c55e, #4ade80)",
            fontSize: "16px",
            fontWeight: 700,
            cursor: greenOrangeDisabled ? "not-allowed" : "pointer",
          }}
        >
          🟩 Green (Deterministic) is Better
        </button>
        <button
          onClick={() => sendMessage({ type: "select_winner", payload: { winner: "trajectory_2" } })}
          disabled={greenOrangeDisabled}
          style={{
            width: "100%",
            marginBottom: "10px",
            borderRadius: "10px",
            padding: "12px",
            border: "none",
            color: "white",
            background: greenOrangeDisabled ? "#9ca3af" : "linear-gradient(90deg, #f97316, #fb923c)",
            fontSize: "16px",
            fontWeight: 700,
            cursor: greenOrangeDisabled ? "not-allowed" : "pointer",
          }}
        >
          🟧 Orange (Stochastic) is Better
        </button>
        <button
          onClick={() => sendMessage({ type: "select_gt_as_winner" })}
          disabled={!status.gt_available}
          style={{
            width: "100%",
            marginBottom: "10px",
            borderRadius: "10px",
            padding: "12px",
            border: "none",
            color: "white",
            background: !status.gt_available ? "#9ca3af" : "linear-gradient(90deg, #dc2626, #ef4444)",
            fontSize: "16px",
            fontWeight: 700,
            cursor: !status.gt_available ? "not-allowed" : "pointer",
          }}
        >
          GT is Best
        </button>
        <button
          onClick={() => sendMessage({ type: "regenerate" })}
          style={{
            width: "100%",
            borderRadius: "10px",
            padding: "12px",
            border: "1px solid #60a5fa",
            color: "#1d4ed8",
            background: "#eff6ff",
            fontSize: "15px",
            fontWeight: 650,
            cursor: "pointer",
          }}
        >
          🔄 Fixed path is better, regenerate random path
        </button>
      </div>
      <div style={{ marginTop: "10px", fontSize: "14px", color: "#4b5563" }}>
        Preferences: {status.total_preferences} / {status.target_count}
      </div>
    </div>
  );
}
