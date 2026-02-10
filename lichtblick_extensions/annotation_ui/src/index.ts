import React from "react";
import { createRoot } from "react-dom/client";
import type { ExtensionContext, PanelExtensionContext } from "@lichtblick/suite";

import { SidebarPanel } from "./panels/SidebarPanel";
import { NavigationPanel } from "./panels/NavigationPanel";
import { ControlsPanel } from "./panels/ControlsPanel";
import { TrajectoryPlotPanel } from "./panels/TrajectoryPlotPanel";
import { VelocityPlotPanel } from "./panels/VelocityPlotPanel";
import { LateralPlotPanel } from "./panels/LateralPlotPanel";
import { MetricsTablePanel } from "./panels/MetricsTablePanel";
import { MetricsCompactPanel } from "./panels/MetricsCompactPanel";
import { SelectionPanel } from "./panels/SelectionPanel";

type PanelComponent = () => JSX.Element;

function mountPanel(Component: PanelComponent) {
  return (context: PanelExtensionContext) => {
    const root = createRoot(context.panelElement);
    root.render(React.createElement(Component));
    return () => root.unmount();
  };
}

export function activate(context: ExtensionContext): void {
  context.registerPanel({ name: "Annotation Sidebar", initPanel: mountPanel(SidebarPanel) });
  context.registerPanel({ name: "Annotation Navigation", initPanel: mountPanel(NavigationPanel) });
  context.registerPanel({ name: "Annotation Controls", initPanel: mountPanel(ControlsPanel) });
  context.registerPanel({ name: "Annotation Trajectory Plot", initPanel: mountPanel(TrajectoryPlotPanel) });
  context.registerPanel({ name: "Annotation Velocity Plot", initPanel: mountPanel(VelocityPlotPanel) });
  context.registerPanel({ name: "Annotation Lateral Plot", initPanel: mountPanel(LateralPlotPanel) });
  context.registerPanel({ name: "Annotation Metrics Table", initPanel: mountPanel(MetricsTablePanel) });
  context.registerPanel({ name: "Annotation Metrics Compact", initPanel: mountPanel(MetricsCompactPanel) });
  context.registerPanel({ name: "Annotation Selection", initPanel: mountPanel(SelectionPanel) });
}
