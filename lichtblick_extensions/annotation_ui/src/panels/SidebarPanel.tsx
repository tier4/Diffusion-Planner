import React from "react";

import { sendMessage } from "../shared/wsClient";
import { useWsState } from "../shared/useWsState";
import { parseHistory, parseProgress, ui } from "../shared/ui";

export function SidebarPanel() {
  const { texts, status, training, isLoading, loadingLabel, lastUpdateNote } = useWsState();
  const progress = parseProgress(texts.progress);
  const history = parseHistory(texts.history);

  return (
    <div style={ui.page}>
      <h3 style={ui.title}>🧭 Annotation Sidebar</h3>

      <div style={ui.section}>
        <h4 style={ui.subtitle}>📈 Status</h4>
        <div
          style={{
            padding: "8px 10px",
            borderRadius: "8px",
            marginBottom: "8px",
            background: isLoading ? "#fff7ed" : "#ecfeff",
            color: isLoading ? "#9a3412" : "#0e7490",
            fontWeight: 700,
            fontSize: "14px",
          }}
        >
          {isLoading ? `⏳ ${loadingLabel ?? "Updating..."}` : `✅ ${lastUpdateNote ?? "Ready"}`}
        </div>
        <div style={{ fontSize: "16px", fontWeight: 600 }}>{progress.sampleText}</div>
        <div style={{ marginTop: "6px", color: "#4b5563", fontSize: "13px", wordBreak: "break-all" }}>
          {progress.fileText}
        </div>
        <div style={{ marginTop: "6px", fontWeight: 500 }}>{progress.preferenceText}</div>
        <div style={{ marginTop: "10px", padding: "8px", background: "#f7fafc", borderRadius: "8px", fontWeight: 600 }}>
          🎯 {texts.metric || "Metric pending..."}
        </div>
        {training && (
          <div style={{ marginTop: "10px", padding: "8px", background: "#eef2ff", borderRadius: "8px" }}>
            <div style={{ fontWeight: 700 }}>🏋️ {training.message}</div>
            <div style={{ fontSize: "13px", color: "#374151", marginTop: "4px" }}>
              Epoch: {training.epoch}/{training.total_epochs} | Batch: {training.batch}/{training.total_batches}
            </div>
            {training.metrics && (
              <div style={{ fontSize: "13px", color: "#374151", marginTop: "2px" }}>
                Loss: {training.metrics.loss?.toFixed?.(4) ?? "-"} | Acc: {training.metrics.accuracy?.toFixed?.(4) ?? "-"}
              </div>
            )}
          </div>
        )}
      </div>

      <div style={ui.section}>
        <h4 style={ui.subtitle}>🗂 Current Sample</h4>
        <div><strong>Index:</strong> {status.current_index + 1} / {status.total_samples}</div>
        <div><strong>Filter:</strong> {status.current_filter}</div>
        <div><strong>Jump Size:</strong> {status.current_jump_size}</div>
        <div><strong>Labeled:</strong> {status.total_preferences} / {status.target_count}</div>
      </div>

      <div style={ui.section}>
        <h4 style={ui.subtitle}>🕘 Recent Labeled</h4>
        {history.length > 0 ? (
          <div style={{ display: "grid", gap: "6px" }}>
            {history.map((item) => (
              <div key={item} style={{ background: "#f9fafb", borderRadius: "8px", padding: "6px 8px" }}>
                ✅ {item}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: "#6b7280" }}>No labels yet.</div>
        )}
      </div>

      <div style={ui.section}>
        <h4 style={ui.subtitle}>🔎 Filters</h4>
        <div>
          {["All", "Finished", "Unfinished"].map((filter) => (
            <label key={filter} style={{ display: "block", marginBottom: "6px", fontSize: "15px" }}>
              <input
                type="radio"
                name="filter"
                checked={status.current_filter === filter}
                onChange={() => sendMessage({ type: "toggle_filter", payload: { filter_mode: filter } })}
              />
              <span style={{ marginLeft: "6px" }}>{filter}</span>
            </label>
          ))}
        </div>

        <label style={{ display: "block", marginTop: "8px", fontSize: "15px" }}>
          <input
            type="checkbox"
            checked={status.auto_skip_labeled}
            onChange={(event) =>
              sendMessage({ type: "set_auto_skip", payload: { enabled: event.target.checked } })
            }
          />
          <span style={{ marginLeft: "6px" }}>⏭ Auto-skip labeled</span>
        </label>
      </div>

      <button
        style={{
          marginTop: "8px",
          width: "100%",
          background: "#ea580c",
          color: "white",
          border: "none",
          borderRadius: "10px",
          padding: "12px",
          fontSize: "16px",
          fontWeight: 700,
          cursor: "pointer",
        }}
        onClick={() => sendMessage({ type: "launch_training" })}
      >
        🚀 Launch Training
      </button>

      {status.annotation_complete && (
        <div style={{ marginTop: "12px", color: "green", fontWeight: 700 }}>✅ Annotation complete</div>
      )}
    </div>
  );
}
