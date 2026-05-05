from typing import Sequence

import cv2
import numpy as np


def _color_from_track_id(track_id):
    if track_id is None:
        return (200, 200, 0)
    value = int(track_id) * 2654435761 % (2**32)
    return (
        80 + (value & 0x7F),
        80 + ((value >> 8) & 0x7F),
        80 + ((value >> 16) & 0x7F),
    )


def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, Sequence) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _clip_line_to_frame(line, width, height):
    x1, y1, x2, y2 = line
    x1 = min(max(int(x1), 0), width - 1)
    x2 = min(max(int(x2), 0), width - 1)
    y1 = min(max(int(y1), 0), height - 1)
    y2 = min(max(int(y2), 0), height - 1)
    return x1, y1, x2, y2


def draw_tracks(frame, boxes, track_ids, class_ids, names, line, total_count, positive_count, negative_count):
    output = frame.copy()
    height, width = output.shape[:2]
    x1, y1, x2, y2 = _clip_line_to_frame(line, width, height)
    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        output,
        f"line count: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"+: {positive_count}  -: {negative_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        left, top, right, bottom = map(int, box)
        color = _color_from_track_id(track_id)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        label = _class_name(names, class_id)
        if track_id is not None:
            label = f"{label} id={track_id}"
        cv2.putText(
            output,
            label,
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
