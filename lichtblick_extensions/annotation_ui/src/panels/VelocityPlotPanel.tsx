import React from "react";

import { useWsState } from "../shared/useWsState";

export function VelocityPlotPanel() {
  const { plots } = useWsState();
  const src = plots.velocity ? `data:image/png;base64,${plots.velocity}` : "";

  return (
    <div style={{ padding: "12px", fontFamily: "sans-serif" }}>
      <h3>Velocity Plot</h3>
      {src ? (
        <img src={src} alt="Velocity plot" style={{ width: "100%" }} />
      ) : (
        <div>No plot available</div>
      )}
    </div>
  );
}
