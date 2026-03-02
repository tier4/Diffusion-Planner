#!/usr/bin/env python3
"""
Convert an Autoware Lanelet2 OSM map to a SUMO net.xml network file.

Uses lanelet2 with MGRSProjector (from the Autoware installation) to load the
map in MGRS local Cartesian coordinates.  The generated net.xml uses
projParameter="!" so SUMO treats the coordinates as-is, giving a 1:1
alignment with the MGRS (x, y) values stored in the .json sidecar files
produced by the data pipeline.

Requires:
    /opt/ros/humble/lib/python3.10/site-packages   (lanelet2)
    /home/danielsanchez/autoware/install/
        autoware_lanelet2_extension_python/
        local/lib/python3.10/dist-packages          (MGRSProjector + reg-elems)

Usage:
    source .venv/bin/activate
    python3 rlvr/scripts/convert_lanelet2_to_sumo.py \\
        --osm /home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm \\
        --output rlvr/sim_config/maps/shinagawa_odaiba.net.xml
"""

import argparse
import json
import math
import sys
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

# Make lanelet2 and autoware_lanelet2_extension_python importable without
# sourcing the full ROS/Autoware workspace.
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")
sys.path.insert(
    0,
    "/home/danielsanchez/autoware/install/"
    "autoware_lanelet2_extension_python/"
    "local/lib/python3.10/dist-packages",
)

import lanelet2  # noqa: E402
from autoware_lanelet2_extension_python.projection import MGRSProjector  # noqa: E402
from lanelet2 import traffic_rules  # noqa: E402
from lanelet2.io import load  # noqa: E402
from lanelet2.routing import RoutingCostDistance, RoutingGraph  # noqa: E402

# Endpoints within this distance (meters) are merged into the same junction.
_ENDPOINT_MERGE_DIST = 2.0
_DEFAULT_SPEED_MPS = 13.89  # 50 km/h fallback
# Small half-size of the junction bounding-box polygon (meters).
_JUNCTION_BOX_R = 0.5


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------


def _load_lanelet_map(osm_path: str):
    """Load the Lanelet2 OSM map with MGRS local Cartesian projection."""
    proj = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    return load(str(osm_path), proj)


def _build_routing_graph(ll_map):
    """Build a routing graph over all lanelets for vehicle traffic."""
    rules = traffic_rules.create(
        traffic_rules.Locations.Germany,
        traffic_rules.Participants.Vehicle,
    )
    cost = RoutingCostDistance(3.0)  # lane-change penalty 3 m
    return RoutingGraph(ll_map, rules, [cost])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _arc_length(centerline) -> float:
    """Total arc length of a lanelet centerline (meters)."""
    total = 0.0
    pts = list(centerline)
    for i in range(1, len(pts)):
        dx = pts[i].x - pts[i - 1].x
        dy = pts[i].y - pts[i - 1].y
        total += math.sqrt(dx * dx + dy * dy)
    return max(total, 0.01)


def _shape_str(centerline) -> str:
    """SUMO shape string: 'x1,y1 x2,y2 ...'"""
    return " ".join(f"{p.x:.4f},{p.y:.4f}" for p in centerline)


def _speed_mps(lanelet) -> float:
    """Extract speed limit (m/s) from lanelet attributes, or fall back to default."""
    attrs = dict(lanelet.attributes)
    if "speed_limit" in attrs:
        try:
            return float(attrs["speed_limit"]) / 3.6
        except ValueError:
            pass
    return _DEFAULT_SPEED_MPS


def _junction_shape(cx: float, cy: float, r: float = _JUNCTION_BOX_R) -> str:
    """Simple axis-aligned box polygon centred on (cx, cy)."""
    return (
        f"{cx - r:.4f},{cy - r:.4f} "
        f"{cx + r:.4f},{cy - r:.4f} "
        f"{cx + r:.4f},{cy + r:.4f} "
        f"{cx - r:.4f},{cy + r:.4f}"
    )


# ---------------------------------------------------------------------------
# Endpoint clustering  →  junction IDs
# ---------------------------------------------------------------------------


def _cluster_endpoints(endpoints: list[tuple[float, float]], threshold: float):
    """
    Assign each (x, y) endpoint to a cluster whose centre is the first point
    within *threshold* metres.  Returns (cluster_ids, cluster_centres).

    cluster_ids:     list[int], one per input point
    cluster_centres: dict[int, (cx, cy)]
    """
    centres: list[tuple[float, float]] = []

    def _find_or_create(x: float, y: float) -> int:
        for i, (cx, cy) in enumerate(centres):
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < threshold:
                return i
        centres.append((x, y))
        return len(centres) - 1

    ids = [_find_or_create(x, y) for (x, y) in endpoints]
    return ids, {i: centres[i] for i in range(len(centres))}


# ---------------------------------------------------------------------------
# net.xml generation
# ---------------------------------------------------------------------------


def _build_net_xml(ll_map, routing_graph) -> ET.Element:
    lanelets = list(ll_map.laneletLayer)

    # --- collect endpoints (start, end) for every lanelet in insertion order ---
    endpoints: list[tuple[float, float]] = []
    for ll in lanelets:
        start = ll.centerline[0]
        end = ll.centerline[-1]
        endpoints.append((start.x, start.y))
        endpoints.append((end.x, end.y))

    endpoint_cluster_ids, cluster_centres = _cluster_endpoints(
        endpoints, _ENDPOINT_MERGE_DIST
    )

    # --- build edge descriptors ---
    # edge_id  →  {from_jn, to_jn, speed, length, shape_str, ll_id}
    # Skip lanelets whose endpoints cluster to the same junction (degenerate edges).
    edges: dict[str, dict] = {}
    ll_id_to_edge: dict[int, str] = {}
    for idx, ll in enumerate(lanelets):
        edge_id = f"ll_{ll.id}"
        from_jn = f"jn_{endpoint_cluster_ids[2 * idx]}"
        to_jn = f"jn_{endpoint_cluster_ids[2 * idx + 1]}"
        if from_jn == to_jn:
            continue  # degenerate edge — start/end within merge threshold
        edges[edge_id] = {
            "from_jn":   from_jn,
            "to_jn":     to_jn,
            "speed":     _speed_mps(ll),
            "length":    _arc_length(ll.centerline),
            "shape_str": _shape_str(ll.centerline),
        }
        ll_id_to_edge[ll.id] = edge_id

    # --- build junction descriptors ---
    # junction_id  →  {x, y, inc_lanes: set}
    junctions: dict[str, dict] = {
        f"jn_{cid}": {"x": cx, "y": cy, "inc_lanes": set()}
        for cid, (cx, cy) in cluster_centres.items()
    }
    for edge_id, info in edges.items():
        junctions[info["to_jn"]]["inc_lanes"].add(f"{edge_id}_0")

    # --- build connection list from the routing graph ---
    connections: list[tuple[str, str]] = []
    for ll in lanelets:
        from_edge = ll_id_to_edge.get(ll.id)
        if from_edge is None:
            continue  # skipped degenerate lanelet
        try:
            following = routing_graph.following(ll)
        except Exception:
            continue
        for fll in following:
            to_edge = ll_id_to_edge.get(fll.id)
            if to_edge:
                connections.append((from_edge, to_edge))

    # --- junction types ---
    # "unregulated" means no right-of-way enforcement and no internal
    # crossing lanes (intLanes=""), which avoids the
    # "invalid logic position" SUMO error that arises when priority
    # junctions have empty intLanes.  This lets us skip the netconvert
    # post-processing step, which collapses curved lane shapes to two
    # endpoints and destroys road geometry accuracy.
    for jn_id, info in junctions.items():
        info["type"] = "unregulated"

    # --- compute network bounding box ---
    all_x = [v["x"] for v in junctions.values()]
    all_y = [v["y"] for v in junctions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    boundary = f"{min_x:.4f},{min_y:.4f},{max_x:.4f},{max_y:.4f}"

    # -----------------------------------------------------------------------
    # Assemble the XML tree
    # -----------------------------------------------------------------------
    root = ET.Element(
        "net",
        attrib={
            "version": "1.16",
            "junctionCornerDetail": "5",
            "limitTurnSpeed": "5.50",
        },
    )

    # <location> — projParameter="!" keeps coordinates in MGRS local Cartesian
    ET.SubElement(
        root,
        "location",
        attrib={
            "netOffset":    "0.00,0.00",
            "convBoundary": boundary,
            "origBoundary": boundary,
            "projParameter": "!",
        },
    )

    # <edge> + <lane>
    for edge_id, info in edges.items():
        edge_el = ET.SubElement(
            root,
            "edge",
            attrib={
                "id":       edge_id,
                "from":     info["from_jn"],
                "to":       info["to_jn"],
                "priority": "-1",
            },
        )
        ET.SubElement(
            edge_el,
            "lane",
            attrib={
                "id":     f"{edge_id}_0",
                "index":  "0",
                "speed":  f"{info['speed']:.4f}",
                "length": f"{info['length']:.4f}",
                "shape":  info["shape_str"],
            },
        )

    # <junction>
    for jn_id, info in junctions.items():
        inc = " ".join(sorted(info["inc_lanes"]))
        ET.SubElement(
            root,
            "junction",
            attrib={
                "id":       jn_id,
                "type":     info["type"],
                "x":        f"{info['x']:.4f}",
                "y":        f"{info['y']:.4f}",
                "incLanes": inc,
                "intLanes": "",
                "shape":    _junction_shape(info["x"], info["y"]),
            },
        )

    # <connection>
    for from_edge, to_edge in connections:
        ET.SubElement(
            root,
            "connection",
            attrib={
                "from":     from_edge,
                "to":       to_edge,
                "fromLane": "0",
                "toLane":   "0",
                "dir":      "s",
                "state":    "M",
            },
        )

    return root, junctions, edges


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Autoware Lanelet2 OSM map to SUMO net.xml."
    )
    parser.add_argument(
        "--osm",
        required=True,
        help="Path to the Lanelet2 OSM file (e.g. lanelet2_map.osm)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the generated SUMO net.xml",
    )
    args = parser.parse_args()

    print(f"Loading lanelet2 map: {args.osm}")
    ll_map = _load_lanelet_map(args.osm)
    print(f"  {len(ll_map.laneletLayer)} lanelets loaded")

    print("Building routing graph…")
    routing_graph = _build_routing_graph(ll_map)

    print("Generating net.xml…")
    net_root, junctions, edges = _build_net_xml(ll_map, routing_graph)

    # Pretty-print with minidom
    xml_str = ET.tostring(net_root, encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")
    # Remove the extra XML declaration added by minidom (we write our own)
    lines = pretty.split("\n")
    lines = [l for l in lines if not l.startswith("<?xml")]
    pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty, encoding="utf-8")

    print(f"  Written {len(junctions)} junctions, {len(edges)} edges")
    print(f"Output: {output_path}")

    # Print one valid edge ID so it can be used in ghost_replay.yaml if needed
    first_edge = next(iter(edges))
    print(f"Example edge ID: {first_edge}  (lane: {first_edge}_0)")


if __name__ == "__main__":
    main()
