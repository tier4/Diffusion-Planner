import React from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";
import { useWsState } from "../shared/useWsState";

export function TrajectoryPlotPanel({ context }: { context: PanelExtensionContext }) {
  const { plots } = useWsState(context);
  const src = plots.trajectory ? `data:image/png;base64,${plots.trajectory}` : "";

  return (
    <div style={{ padding: "12px", fontFamily: "sans-serif" }}>
      <h3>Trajectory Plot</h3>
      {src ? (
        <img src={src} alt="Trajectory plot" style={{ width: "100%" }} />
      ) : (
        <div>No plot available</div>
      )}
    </div>
  );
}
