import argparse
from pathlib import Path

import lanelet2
import matplotlib.pyplot as plt
import numpy as np
from autoware_lanelet2_extension_python.projection import MGRSProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("lanelet_path", type=Path)
    parser.add_argument(
        "--crop_bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop bounding box: X1 Y1 X2 Y2 (meters)",
        default=None,
    )
    return parser.parse_args()


def _get_attribute(attribute_map, key: str, default: str) -> str:
    if key in attribute_map:
        return attribute_map[key]
    else:
        return default


if __name__ == "__main__":
    args = parse_args()
    lanelet_path = args.lanelet_path
    crop_bbox = args.crop_bbox
    print(lanelet_path)

    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(str(lanelet_path), projection)

    # check attributes
    print(f"{len(lanelet_map.laneletLayer)=}")
    print(f"{len(lanelet_map.areaLayer)=}")
    print(f"{len(lanelet_map.regulatoryElementLayer)=}")
    print(f"{len(lanelet_map.polygonLayer)=}")
    print(f"{len(lanelet_map.lineStringLayer)=}")
    print(f"{len(lanelet_map.pointLayer)=}")

    num_pedestrian_lane = 0

    array_pedestrian_lanes = []
    array_other_lanes = []

    for lanelet in lanelet_map.laneletLayer:
        lanelet_subtype = _get_attribute(lanelet.attributes, "subtype", "")
        is_pedestrian_lane = lanelet_subtype == "pedestrian_lane"
        color = "red" if is_pedestrian_lane else "black"
        num_pedestrian_lane += is_pedestrian_lane

        xyz = np.array([[p.x, p.y, p.z] for p in lanelet.centerline])

        if crop_bbox is not None:
            x1, y1, x2, y2 = crop_bbox
            mask = (xyz[:, 0] >= x1) & (xyz[:, 0] <= x2) & (xyz[:, 1] >= y1) & (xyz[:, 1] <= y2)
            xyz = xyz[mask]
            if len(xyz) == 0:
                continue

        if is_pedestrian_lane:
            array_pedestrian_lanes.append(xyz)
        else:
            array_other_lanes.append(xyz)

    array_pedestrian_lanes = np.vstack(array_pedestrian_lanes)
    array_other_lanes = np.vstack(array_other_lanes)

    plt.plot(
        array_other_lanes[:, 0], array_other_lanes[:, 1], ".", color="black", label="other lanes"
    )
    plt.plot(
        array_pedestrian_lanes[:, 0],
        array_pedestrian_lanes[:, 1],
        ".",
        color="red",
        label="pedestrian lanes",
    )

    print(f"num_pedestrian_lane: {num_pedestrian_lane}")
    plt.legend()
    plt.axis("equal")
    plt.savefig("out.png")
