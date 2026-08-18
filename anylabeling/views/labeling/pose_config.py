"""Helpers for project-scoped YOLO pose annotation configs."""

import os

import yaml


class PoseConfigError(ValueError):
    """Raised when a pose config cannot be used for annotation."""


def _clean_names(names, context):
    clean_names = []
    seen = set()
    duplicates = []
    for name in names or []:
        clean_name = str(name or "").strip()
        if not clean_name:
            continue
        key = clean_name.casefold()
        if key in seen:
            duplicates.append(clean_name)
            continue
        seen.add(key)
        clean_names.append(clean_name)
    if duplicates:
        raise PoseConfigError(
            f"{context} has duplicate names: {', '.join(sorted(duplicates))}"
        )
    return clean_names


def _normalize_skeleton_edges(
    edges, class_name, keypoints, context, strict=True
):
    if edges is None:
        return []
    if not isinstance(edges, list):
        if not strict:
            return []
        raise PoseConfigError(f"{context} must be a list.")

    valid_keypoints = set(keypoints)
    normalized = []
    seen = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            if not strict:
                continue
            raise PoseConfigError(f"{context} has an invalid edge: {edge}")
        start = str(edge[0] or "").strip()
        end = str(edge[1] or "").strip()
        if not start or not end:
            if not strict:
                continue
            raise PoseConfigError(f"{context} has an empty edge endpoint.")
        if start not in valid_keypoints or end not in valid_keypoints:
            if not strict:
                continue
            raise PoseConfigError(
                f"{context} for class {class_name} references unknown "
                f"keypoints: {start}, {end}"
            )
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append([start, end])
    return normalized


def _validate_top_level_skeleton_edges(edges, classes):
    if not edges:
        return
    if not isinstance(edges, list):
        raise PoseConfigError("skeleton must be a list.")

    class_keypoints = [set(keypoints) for keypoints in classes.values()]
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise PoseConfigError(f"skeleton has an invalid edge: {edge}")
        start = str(edge[0] or "").strip()
        end = str(edge[1] or "").strip()
        if not start or not end:
            raise PoseConfigError("skeleton has an empty edge endpoint.")
        if not any(
            start in keypoints and end in keypoints
            for keypoints in class_keypoints
        ):
            raise PoseConfigError(
                "skeleton references unknown keypoints: "
                f"{start}, {end}"
            )


def _normalize_skeletons(config, classes):
    raw_skeleton = config.get("skeleton") or []
    raw_skeletons = config.get("skeletons")
    if raw_skeletons is None:
        raw_skeletons = {}
    if not isinstance(raw_skeletons, dict):
        raise PoseConfigError("skeletons must be a dict.")
    for class_name in raw_skeletons:
        if class_name not in classes:
            raise PoseConfigError(
                f"skeletons references unknown class: {class_name}"
            )

    _validate_top_level_skeleton_edges(raw_skeleton, classes)
    skeletons = {}
    for class_name, keypoints in classes.items():
        class_edges = raw_skeletons.get(class_name, raw_skeleton)
        skeletons[class_name] = _normalize_skeleton_edges(
            class_edges,
            class_name,
            keypoints,
            f"skeletons.{class_name}",
            strict=class_name in raw_skeletons,
        )
    return skeletons


def _normalize_template_points(template_points, classes):
    template_points = template_points or {}
    if not isinstance(template_points, dict):
        raise PoseConfigError("template_points must be a dict.")
    for class_name in template_points:
        if class_name not in classes:
            raise PoseConfigError(
                f"template_points references unknown class: {class_name}"
            )

    normalized = {}
    for class_name, keypoints in classes.items():
        raw_template = template_points.get(class_name) or {}
        if not isinstance(raw_template, dict):
            raise PoseConfigError(
                f"template_points.{class_name} must be a dict."
            )
        valid_keypoints = set(keypoints)
        class_template = {}
        for keypoint_name, point in raw_template.items():
            clean_keypoint_name = str(keypoint_name or "").strip()
            if clean_keypoint_name not in valid_keypoints:
                raise PoseConfigError(
                    f"template_points.{class_name} references unknown "
                    f"keypoint: {clean_keypoint_name}"
                )
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise PoseConfigError(
                    f"template_points.{class_name}.{clean_keypoint_name} "
                    "must be [x, y]."
                )
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError) as exc:
                raise PoseConfigError(
                    f"template_points.{class_name}.{clean_keypoint_name} "
                    "must contain numbers."
                ) from exc
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise PoseConfigError(
                    f"template_points.{class_name}.{clean_keypoint_name} "
                    "must be within 0 and 1."
                )
            class_template[clean_keypoint_name] = [x, y]
        normalized[class_name] = class_template
    return normalized


def normalize_pose_config(config):
    """Normalize a project pose config into the schema used by the UI."""
    config = config or {}
    classes = config.get("classes") or {}
    if isinstance(classes, list):
        classes = {"person": classes}
    if not isinstance(classes, dict) or not classes:
        raise PoseConfigError("pose config requires classes.")

    normalized_classes = {}
    for class_name, keypoints in classes.items():
        clean_class_name = str(class_name or "").strip()
        if not clean_class_name:
            raise PoseConfigError("pose config contains an empty class name.")
        clean_keypoints = _clean_names(
            keypoints, f"classes.{clean_class_name}"
        )
        if not clean_keypoints:
            raise PoseConfigError(
                f"classes.{clean_class_name} requires keypoints."
            )
        normalized_classes[clean_class_name] = clean_keypoints

    keypoint_counts = {len(value) for value in normalized_classes.values()}
    if len(keypoint_counts) != 1:
        raise PoseConfigError(
            "all pose classes must have the same keypoint count."
        )
    keypoint_count = keypoint_counts.pop()

    flip_idx = config.get("flip_idx") or []
    if flip_idx:
        try:
            flip_idx = [int(index) for index in flip_idx]
        except (TypeError, ValueError) as exc:
            raise PoseConfigError("flip_idx must contain integers.") from exc
        if len(flip_idx) != keypoint_count:
            raise PoseConfigError(
                "flip_idx length must equal the keypoint count."
            )
        if sorted(flip_idx) != list(range(keypoint_count)):
            raise PoseConfigError(
                "flip_idx must cover every keypoint index once."
            )

    skeletons = _normalize_skeletons(config, normalized_classes)
    first_class_name = next(iter(normalized_classes))
    return {
        "classes": normalized_classes,
        "skeletons": skeletons,
        "skeleton": [edge[:] for edge in skeletons.get(first_class_name, [])],
        "template_points": _normalize_template_points(
            config.get("template_points"), normalized_classes
        ),
        "flip_idx": flip_idx,
        "has_visible": bool(config.get("has_visible", True)),
    }


def load_pose_config(path):
    """Load and normalize a pose config, returning None for empty paths."""
    if not path:
        return None
    if not os.path.exists(path):
        raise PoseConfigError(f"pose config does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return normalize_pose_config(yaml.safe_load(f) or {})


def class_names(config):
    return list((config or {}).get("classes", {}).keys())


def rectangle_from_points(points):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def rectangle_points(rect):
    left, top, right, bottom = rect
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ]


def template_points_for_bbox(config, class_name, rect):
    keypoints = (config or {}).get("classes", {}).get(class_name, [])
    template = (
        (config or {}).get("template_points", {}).get(class_name, {}) or {}
    )
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    points = {}
    keypoint_count = max(len(keypoints), 1)
    for index, keypoint_name in enumerate(keypoints):
        rel_x, rel_y = template.get(
            keypoint_name, [0.5, (index + 1) / (keypoint_count + 1)]
        )
        points[keypoint_name] = [
            left + width * float(rel_x),
            top + height * float(rel_y),
        ]
    return points


def create_pose_shape_dicts(config, class_name, bbox_points, group_id):
    """Create LabelMe-compatible rectangle and point dictionaries."""
    if class_name not in (config or {}).get("classes", {}):
        raise PoseConfigError(f"unknown pose class: {class_name}")

    rect = rectangle_from_points(bbox_points)
    shapes = [
        {
            "label": class_name,
            "points": rectangle_points(rect),
            "group_id": group_id,
            "shape_type": "rectangle",
            "flags": {},
        }
    ]
    for keypoint_name, point in template_points_for_bbox(
        config, class_name, rect
    ).items():
        shapes.append(
            {
                "label": keypoint_name,
                "points": [point],
                "group_id": group_id,
                "shape_type": "point",
                "flags": {},
            }
        )
    return shapes


def map_point_between_rects(point, old_rect, new_rect):
    old_left, old_top, old_right, old_bottom = old_rect
    new_left, new_top, new_right, new_bottom = new_rect
    old_width = old_right - old_left
    old_height = old_bottom - old_top
    new_width = new_right - new_left
    new_height = new_bottom - new_top

    if old_width:
        rel_x = (point[0] - old_left) / old_width
        x = new_left + rel_x * new_width
    else:
        x = point[0] + (new_left - old_left)

    if old_height:
        rel_y = (point[1] - old_top) / old_height
        y = new_top + rel_y * new_height
    else:
        y = point[1] + (new_top - old_top)
    return [x, y]


def skeleton_segments(shapes, config):
    """Return drawable skeleton line data for existing Shape objects."""
    if not config:
        return []

    grouped = {}
    for shape in shapes:
        group_id = getattr(shape, "group_id", None)
        if group_id is None or not getattr(shape, "visible", True):
            continue
        grouped.setdefault(group_id, []).append(shape)

    segments = []
    for group_id, group_shapes in grouped.items():
        rectangles = [
            shape
            for shape in group_shapes
            if getattr(shape, "shape_type", None) == "rectangle"
        ]
        if not rectangles:
            continue
        class_name = getattr(rectangles[0], "label", "")
        edges = (config.get("skeletons") or {}).get(class_name, [])
        if not edges:
            continue
        points = {
            getattr(shape, "label", ""): shape.points[0]
            for shape in group_shapes
            if getattr(shape, "shape_type", None) == "point"
            and getattr(shape, "points", None)
        }
        for start, end in edges:
            if start not in points or end not in points:
                continue
            segments.append(
                {
                    "group_id": group_id,
                    "class_name": class_name,
                    "start_label": start,
                    "end_label": end,
                    "start": points[start],
                    "end": points[end],
                }
            )
    return segments
