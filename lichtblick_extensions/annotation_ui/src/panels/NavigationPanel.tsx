import React, { useEffect, useState } from "react";

import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { ui } from "../shared/ui";

export function NavigationPanel() {
  const { status } = useWsState();
  const [jumpIndex, setJumpIndex] = useState<number>(1);

  const jumpButtons = [
    { delta: -30, label: "← 30" },
    { delta: -10, label: "← 10" },
    { delta: -1, label: "← 1" },
    { delta: 1, label: "1 →" },
    { delta: 10, label: "10 →" },
    { delta: 30, label: "30 →" },
  ];

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
        return;
      }
      const jumpSize = Math.max(1, Number(status.current_jump_size || 1));
      const delta = event.key === "ArrowLeft" ? -jumpSize : jumpSize;
      sendMessage({ type: "jump", payload: { delta } });
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [status.current_jump_size]);

  return (
    <div style={{ ...ui.page, textAlign: "center" }}>
      <h3 style={ui.title}>🧭 Navigation</h3>
      <div style={{ ...ui.section, display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "center" }}>
        {jumpButtons.map((button) => (
          <button
            key={button.delta}
            onClick={() => sendMessage({ type: "jump", payload: { delta: button.delta } })}
            style={{
              minWidth: "92px",
              padding: "10px 12px",
              borderRadius: "10px",
              border: "1px solid #d1d5db",
              fontSize: "16px",
              fontWeight: 700,
              color: button.delta < 0 ? "#b91c1c" : "#166534",
              background: button.delta < 0 ? "#fef2f2" : "#f0fdf4",
              cursor: "pointer",
            }}
          >
            {button.label}
          </button>
        ))}
      </div>

      <div style={{ ...ui.section, marginTop: "8px" }}>
        <label style={{ fontSize: "16px", fontWeight: 600 }}>
          🔢 Jump to sample
          <input
            type="number"
            min={1}
            value={jumpIndex}
            onChange={(event) => setJumpIndex(Number(event.target.value))}
            style={{ marginLeft: "8px", width: "100px", padding: "6px", fontSize: "15px" }}
          />
        </label>
        <button
          style={{ marginLeft: "8px", padding: "8px 12px", borderRadius: "8px", border: "1px solid #d1d5db" }}
          onClick={() => sendMessage({ type: "jump_to_index", payload: { target_index: jumpIndex } })}
        >
          Go 🚀
        </button>
      </div>

      <button
        style={{ marginTop: "6px", padding: "10px 14px", borderRadius: "10px", border: "1px solid #3b82f6", color: "#1d4ed8" }}
        onClick={() => sendMessage({ type: "jump_to_next_unlabeled" })}
      >
        ⏭️ Next Unlabeled
      </button>

      <div style={{ marginTop: "12px", fontSize: "14px", color: "#4b5563" }}>
        Current index: {status.current_index + 1} / {status.total_samples}
      </div>
      <div style={{ marginTop: "4px", fontSize: "13px", color: "#6b7280" }}>
        Keyboard: ← / → uses current jump size ({status.current_jump_size})
      </div>
    </div>
  );
}
