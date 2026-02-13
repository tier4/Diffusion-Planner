import React from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { useWsState } from "../shared/useWsState";

export function MetricsCompactPanel({ context }: { context: PanelExtensionContext }) {
  const { texts } = useWsState(context);

  return (
    <div style={{ padding: "12px", fontFamily: "sans-serif" }}>
      <h3>Metrics (ADE/FDE)</h3>
      <pre style={{ whiteSpace: "pre-wrap" }}>{texts.metrics_ade_fde_table}</pre>
    </div>
  );
}
