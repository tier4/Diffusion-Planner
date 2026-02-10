import React from "react";

import { useWsState } from "../shared/useWsState";

export function LateralPlotPanel() {
  const { plots } = useWsState();
  const src = plots.lateral ? `data:image/png;base64,${plots.lateral}` : "";

  return (
    <div style={{ padding: "12px", fontFamily: "sans-serif" }}>
      <h3>Lateral/Curvature Plot</h3>
      {src ? (
        <img src={src} alt="Lateral plot" style={{ width: "100%" }} />
      ) : (
        <div>No plot available</div>
      )}
    </div>
  );
}
