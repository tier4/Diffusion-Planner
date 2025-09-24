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

    lanelet_type_set = set()
    lanelet_subtype_set = set()
    centerline_type_set = set()
    centerline_subtype_set = set()
    boundary_line_type_set = set()
    boundary_line_subtype_set = set()
    area_type_set = set()
    area_subtype_set = set()
    regulatory_type_set = set()
    regulatory_subtype_set = set()
    polygon_type_set = set()
    polygon_subtype_set = set()
    line_string_type_set = set()
    line_string_subtype_set = set()
    point_type_set = set()
    point_subtype_set = set()
    for search_root in search_roots:
        lanelet_path_list = sorted(search_root.rglob("*.osm"))

        for lanelet_path in lanelet_path_list:
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

            for lanelet in lanelet_map.laneletLayer:
                lanelet_type = _get_attribute(lanelet.attributes, "type", "")
                if lanelet_type != "":
                    lanelet_type_set.add(lanelet_type)

                lanelet_subtype = _get_attribute(lanelet.attributes, "subtype", "")
                if lanelet_subtype != "":
                    lanelet_subtype_set.add(lanelet_subtype)

                centerline_type = _get_attribute(lanelet.centerline.attributes, "type", "")
                if centerline_type != "":
                    centerline_type_set.add(centerline_type)
                centerline_subtype = _get_attribute(lanelet.centerline.attributes, "subtype", "")
                if centerline_subtype != "":
                    centerline_subtype_set.add(centerline_subtype)

                line_type_left = _get_attribute(lanelet.leftBound.attributes, "type", "")
                if line_type_left != "":
                    boundary_line_type_set.add(line_type_left)
                line_type_left_subtype = _get_attribute(lanelet.leftBound.attributes, "subtype", "")
                if line_type_left_subtype != "":
                    boundary_line_subtype_set.add(line_type_left_subtype)

                line_type_right = _get_attribute(lanelet.rightBound.attributes, "type", "")
                if line_type_right != "":
                    boundary_line_type_set.add(line_type_right)
                line_type_right_subtype = _get_attribute(
                    lanelet.rightBound.attributes, "subtype", ""
                )
                if line_type_right_subtype != "":
                    boundary_line_subtype_set.add(line_type_right_subtype)

            for area in lanelet_map.areaLayer:
                area_type = _get_attribute(area.attributes, "type", "")
                if area_type != "":
                    area_type_set.add(area_type)
                area_subtype = _get_attribute(area.attributes, "subtype", "")
                if area_subtype != "":
                    area_subtype_set.add(area_subtype)

            for regulatory in lanelet_map.regulatoryElementLayer:
                reg_type = _get_attribute(regulatory.attributes, "type", "")
                if reg_type != "":
                    regulatory_type_set.add(reg_type)
                reg_subtype = _get_attribute(regulatory.attributes, "subtype", "")
                if reg_subtype != "":
                    regulatory_subtype_set.add(reg_subtype)

            for polygon in lanelet_map.polygonLayer:
                poly_type = _get_attribute(polygon.attributes, "type", "")
                if poly_type != "":
                    polygon_type_set.add(poly_type)
                poly_subtype = _get_attribute(polygon.attributes, "subtype", "")
                if poly_subtype != "":
                    polygon_subtype_set.add(poly_subtype)

            for line in lanelet_map.lineStringLayer:
                boundary_line_type = _get_attribute(line.attributes, "type", "")
                if boundary_line_type != "":
                    line_string_type_set.add(boundary_line_type)
                boundary_line_subtype = _get_attribute(line.attributes, "subtype", "")
                if boundary_line_subtype != "":
                    line_string_subtype_set.add(boundary_line_subtype)

            for point in lanelet_map.pointLayer:
                point_type = _get_attribute(point.attributes, "type", "")
                if point_type != "":
                    point_type_set.add(point_type)
                point_subtype = _get_attribute(point.attributes, "subtype", "")
                if point_subtype != "":
                    point_subtype_set.add(point_subtype)

    print("Unique lanelet types found:")
    lanelet_type_list = sorted(lanelet_type_set)
    for i, lanelet_type in enumerate(lanelet_type_list):
        print(f" - {i}: {lanelet_type}")
    print("Unique lanelet subtypes found:")
    lanelet_subtype_list = sorted(lanelet_subtype_set)
    for i, lanelet_subtype in enumerate(lanelet_subtype_list):
        print(f" - {i}: {lanelet_subtype}")

    print("Unique centerline types found:")
    centerline_type_list = sorted(centerline_type_set)
    for i, centerline_type in enumerate(centerline_type_list):
        print(f" - {i}: {centerline_type}")
    print("Unique centerline subtypes found:")
    centerline_subtype_list = sorted(centerline_subtype_set)
    for i, centerline_subtype in enumerate(centerline_subtype_list):
        print(f" - {i}: {centerline_subtype}")

    print("Unique boundary line types found:")
    boundary_line_type_list = sorted(boundary_line_type_set)
    for i, boundary_line_type in enumerate(boundary_line_type_list):
        print(f" - {i}: {boundary_line_type}")
    print("Unique boundary line subtypes found:")
    boundary_line_subtype_list = sorted(boundary_line_subtype_set)
    for i, boundary_line_subtype in enumerate(boundary_line_subtype_list):
        print(f" - {i}: {boundary_line_subtype}")

    print("Unique area types found:")
    area_type_list = sorted(area_type_set)
    for i, area_type in enumerate(area_type_list):
        print(f" - {i}: {area_type}")
    print("Unique area subtypes found:")
    area_subtype_list = sorted(area_subtype_set)
    for i, area_subtype in enumerate(area_subtype_list):
        print(f" - {i}: {area_subtype}")

    print("Unique regulatory types found:")
    regulatory_type_list = sorted(regulatory_type_set)
    for i, regulatory_type in enumerate(regulatory_type_list):
        print(f" - {i}: {regulatory_type}")
    print("Unique regulatory subtypes found:")
    regulatory_subtype_list = sorted(regulatory_subtype_set)
    for i, regulatory_subtype in enumerate(regulatory_subtype_list):
        print(f" - {i}: {regulatory_subtype}")

    print("Unique polygon types found:")
    polygon_type_list = sorted(polygon_type_set)
    for i, polygon_type in enumerate(polygon_type_list):
        print(f" - {i}: {polygon_type}")
    print("Unique polygon subtypes found:")
    polygon_subtype_list = sorted(polygon_subtype_set)
    for i, polygon_subtype in enumerate(polygon_subtype_list):
        print(f" - {i}: {polygon_subtype}")

    print("Unique line string types found:")
    line_string_type_list = sorted(line_string_type_set)
    for i, line_string_type in enumerate(line_string_type_list):
        print(f" - {i}: {line_string_type}")
    print("Unique line string subtypes found:")
    line_string_subtype_list = sorted(line_string_subtype_set)
    for i, line_string_subtype in enumerate(line_string_subtype_list):
        print(f" - {i}: {line_string_subtype}")

    print("Unique point types found:")
    point_type_list = sorted(point_type_set)
    for i, point_type in enumerate(point_type_list):
        print(f" - {i}: {point_type}")
    print("Unique point subtypes found:")
    point_subtype_list = sorted(point_subtype_set)
    for i, point_subtype in enumerate(point_subtype_list):
        print(f" - {i}: {point_subtype}")
