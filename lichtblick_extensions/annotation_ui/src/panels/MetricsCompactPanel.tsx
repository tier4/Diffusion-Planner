import React from "react";

import { useWsState } from "../shared/useWsState";

export function MetricsCompactPanel() {
  const { texts } = useWsState();

  return (
    <div style={{ padding: "12px", fontFamily: "sans-serif" }}>
      <h3>Metrics (ADE/FDE)</h3>
      <pre style={{ whiteSpace: "pre-wrap" }}>{texts.metrics_ade_fde_table}</pre>
    </div>
  );
}
