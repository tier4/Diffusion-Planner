import React from "react";

import { useWsState } from "../shared/useWsState";
import { parseMarkdownTable, ui } from "../shared/ui";

export function MetricsTablePanel() {
  const { texts } = useWsState();
  const table = parseMarkdownTable(texts.metrics_full_table || texts.metrics);
  const betterRow = (row: string[], idx: number) => {
    const metric = row[0] ?? "";
    const greenVal = row[1] ?? "";
    const orangeVal = row[2] ?? "";
    if (idx === 0) {
      return { green: false, orange: false };
    }
    if (metric.toLowerCase().includes("lower")) {
      return { green: false, orange: false };
    }
    return {
      green: greenVal.includes("✓"),
      orange: orangeVal.includes("✓"),
    };
  };

  return (
    <div style={ui.page}>
      <h3 style={ui.title}>📊 Metrics Table</h3>
      <div style={{ ...ui.section, overflowX: "auto" }}>
        {table.headers.length > 0 ? (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
            <thead>
              <tr>
                {table.headers.map((header) => (
                  <th
                    key={header}
                    style={{
                      borderBottom: "2px solid #d1d5db",
                      textAlign: "left",
                      padding: "8px",
                      fontSize: "15px",
                      color:
                        header.toLowerCase().includes("green")
                          ? "#77B680"
                          : header.toLowerCase().includes("orange")
                            ? "#ee5912"
                            : "#111827",
                    }}
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, idx) => (
                <tr key={`${row[0]}-${idx}`}>
                  {row.map((cell, cellIdx) => {
                    const better = betterRow(row, idx);
                    const isGreenCol = cellIdx === 1;
                    const isOrangeCol = cellIdx === 2;
                    const highlight = (isGreenCol && better.green) || (isOrangeCol && better.orange);
                    return (
                      <td
                        key={`${cell}-${cellIdx}`}
                        style={{
                          borderBottom: "1px solid #e5e7eb",
                          padding: "8px",
                          fontWeight: highlight ? 700 : 500,
                          color: highlight ? (isGreenCol ? "#77B680" : "#ee5912") : "#111827",
                        }}
                      >
                      {cell}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ color: "#6b7280" }}>Metrics are not available yet.</div>
        )}
      </div>
    </div>
  );
}
