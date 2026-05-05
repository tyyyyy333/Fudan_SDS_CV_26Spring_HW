import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from hw2_cv.utils import ensure_dir, log_info, save_json, save_yaml


VISDRONE_CATEGORY_TO_YOLO = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}


def _match_split(name):
    normalized = name.lower()
    if "train" in normalized:
        return "train"
    if "val" in normalized:
        return "val"
    if "test" in normalized:
        return "test"
    return None


def _discover_split_dirs(raw_root):
    split_dirs = {}
    candidates = [raw_root]
    candidates.extend(path for path in raw_root.iterdir() if path.is_dir())

    for candidate in candidates:
        split_name = _match_split(candidate.name)
        if split_name is None or split_name in split_dirs:
            continue
        images_dir = candidate / "images"
        if images_dir.exists() and images_dir.is_dir():
            split_dirs[split_name] = candidate

    if not split_dirs:
        raise FileNotFoundError(f"No VisDrone split directories found under {raw_root}.")
    return split_dirs


def _iter_images(images_dir):
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def _materialize_image(source, destination, link_mode):
    if destination.exists():
        destination.unlink()

    if link_mode == "copy":
        shutil.copy2(source, destination)
        return
    if link_mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    raise ValueError(f"Unsupported link mode: {link_mode}")


def _normalize_box(bbox_left, bbox_top, bbox_width, bbox_height, image_width, image_height):
    x1 = max(0.0, bbox_left)
    y1 = max(0.0, bbox_top)
    x2 = min(float(image_width), bbox_left + bbox_width)
    y2 = min(float(image_height), bbox_top + bbox_height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 1.0 or clipped_height <= 1.0:
        return None

    center_x = (x1 + x2) * 0.5 / image_width
    center_y = (y1 + y2) * 0.5 / image_height
    width = clipped_width / image_width
    height = clipped_height / image_height
    return center_x, center_y, width, height


def _convert_annotation(annotation_path, image_width, image_height, category_mapping):
    label_lines = []
    class_counts = {}

    if annotation_path is None or not annotation_path.exists():
        return label_lines, class_counts

    with annotation_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue

            values = stripped.split(",")
            if len(values) < 6:
                continue

            bbox_left, bbox_top, bbox_width, bbox_height = map(float, values[:4])
            valid_flag = int(float(values[4]))
            category_id = int(float(values[5]))
            if valid_flag <= 0:
                continue
            if category_id not in category_mapping:
                continue

            normalized_box = _normalize_box(
                bbox_left=bbox_left,
                bbox_top=bbox_top,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                image_width=image_width,
                image_height=image_height,
            )
            if normalized_box is None:
                continue

            class_id = category_mapping[category_id]
            center_x, center_y, width, height = normalized_box
            label_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            )
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return label_lines, class_counts


def _split_summary_template():
    return {
        "images": 0,
        "labeled_images": 0,
        "empty_images": 0,
        "boxes": 0,
        "class_counts": {},
    }


def convert_visdrone_to_yolo(raw_root, output_root, link_mode="copy", category_mapping=None, data_yaml_path=None):
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    category_mapping = category_mapping or VISDRONE_CATEGORY_TO_YOLO
    split_dirs = _discover_split_dirs(raw_root)
    log_info(f"[visdrone] converting {raw_root} -> {output_root}")

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "link_mode": link_mode,
        "splits": {},
    }

    for split_name, split_root in split_dirs.items():
        images_dir = split_root / "images"
        annotations_dir = split_root / "annotations"
        target_images_dir = ensure_dir(output_root / "images" / split_name)
        target_labels_dir = ensure_dir(output_root / "labels" / split_name)
        split_summary = _split_summary_template()
        image_paths = _iter_images(images_dir)
        log_info(f"[visdrone] split={split_name} | images={len(image_paths)}")

        for image_path in tqdm(image_paths, desc=f"visdrone {split_name}", leave=False):
            target_image_path = target_images_dir / image_path.name
            target_label_path = target_labels_dir / f"{image_path.stem}.txt"
            annotation_path = annotations_dir / f"{image_path.stem}.txt"
            if not annotation_path.exists():
                annotation_path = None

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            label_lines, class_counts = _convert_annotation(
                annotation_path=annotation_path,
                image_width=image_width,
                image_height=image_height,
                category_mapping=category_mapping,
            )

            _materialize_image(image_path, target_image_path, link_mode=link_mode)
            target_label_path.write_text("\n".join(label_lines), encoding="utf-8")

            split_summary["images"] += 1
            split_summary["boxes"] += len(label_lines)
            if label_lines:
                split_summary["labeled_images"] += 1
            else:
                split_summary["empty_images"] += 1

            for class_id, count in class_counts.items():
                split_summary["class_counts"][str(class_id)] = (
                    split_summary["class_counts"].get(str(class_id), 0) + count
                )

        summary["splits"][split_name] = split_summary
        log_info(
            f"[visdrone] split={split_name} done | boxes={split_summary['boxes']} | "
            f"labeled={split_summary['labeled_images']} | empty={split_summary['empty_images']}"
        )

    if data_yaml_path is not None:
        save_yaml(
            {
                "path": str(output_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    0: "pedestrian",
                    1: "people",
                    2: "bicycle",
                    3: "car",
                    4: "van",
                    5: "truck",
                    6: "tricycle",
                    7: "awning-tricycle",
                    8: "bus",
                    9: "motor",
                },
            },
            data_yaml_path,
        )
        summary["data_yaml_path"] = str(data_yaml_path)

    save_json(summary, output_root / "conversion_summary.json")
    log_info("[visdrone] conversion finished")
    return summary
