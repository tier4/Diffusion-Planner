import React from "react";

import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { ui } from "../shared/ui";

export function SelectionPanel() {
  const { status } = useWsState();

  return (
    <div style={ui.page}>
      <h3 style={ui.title}>✅ Annotation Selection</h3>
      <div style={ui.section}>
        <button
          onClick={() => sendMessage({ type: "select_winner", payload: { winner: "trajectory_2" } })}
          style={{
            width: "100%",
            marginBottom: "10px",
            borderRadius: "10px",
            padding: "12px",
            border: "none",
            color: "white",
            background: "linear-gradient(90deg, #f97316, #fb923c)",
            fontSize: "16px",
            fontWeight: 700,
            cursor: "pointer",
          }}
        >
          🟧 Orange (Stochastic) is Better
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
