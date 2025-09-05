import argparse
from pathlib import Path

import lanelet2
from autoware_lanelet2_extension_python.projection import MGRSProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_roots", type=Path, nargs="+")
    return parser.parse_args()


def _get_attribute(attribute_map, key: str, default: str) -> str:
    """Return attribute value from AttributeMap with default fallback.

    Args:
    ----
        attribute_map: AttributeMap object.
        key (str): Attribute key to retrieve.
        default (str): Default value if key is not found.

    Returns:
    -------
        str: Attribute value or default if key is not found.

    """
    if key in attribute_map:
        return attribute_map[key]
    else:
        return default


if __name__ == "__main__":
    args = parse_args()
    search_roots = args.search_roots

    line_type_set = set()
    for search_root in search_roots:
        lanelet_path_list = sorted(search_root.rglob("*.osm"))

        for lanelet_path in lanelet_path_list:
            print(lanelet_path)
            projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
            lanelet_map = lanelet2.io.load(str(lanelet_path), projection)

            for lanelet in lanelet_map.laneletLayer:
                line_type_left = _get_attribute(lanelet.leftBound.attributes, "type", "")
                line_type_set.add(line_type_left)

                line_type_right = _get_attribute(lanelet.rightBound.attributes, "type", "")
                line_type_set.add(line_type_right)

    print("Unique line types found:")
    line_type_list = sorted(line_type_set)
    for i, line_type in enumerate(line_type_list):
        print(f" - {i}: {line_type}")
