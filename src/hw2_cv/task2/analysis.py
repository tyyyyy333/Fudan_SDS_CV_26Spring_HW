def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def summarize_tracks(track_states):
    rows = []
    for track_id in sorted(track_states):
        state = track_states[track_id]
        dominant_class_id = max(state["class_counts"], key=state["class_counts"].get)
        rows.append(
            {
                "track_id": track_id,
                "first_frame": state["first_frame"],
                "last_frame": state["last_frame"],
                "frames_seen": state["frames_seen"],
                "dominant_class_id": dominant_class_id,
                "class_counts": dict(sorted(state["class_counts"].items())),
            }
        )
    return rows


def _analyze_record_pairs(records, iou_threshold=0.5):
    kept_events = []
    switched_events = []
    lost_events = []

    for current_record, next_record in zip(records, records[1:]):
        used_next_indices = set()
        for current_det in current_record["detections"]:
            best_match = None
            best_iou = 0.0
            for next_index, next_det in enumerate(next_record["detections"]):
                if next_index in used_next_indices:
                    continue
                if current_det["class_id"] != next_det["class_id"]:
                    continue
                current_track_id = current_det["track_id"]
                next_track_id = next_det["track_id"]
                if current_track_id is None or next_track_id is None:
                    continue
                iou = box_iou(current_det["box"], next_det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = (next_index, next_det)

            if best_match is None or best_iou < iou_threshold:
                lost_events.append(
                    {
                        "from_frame": current_record["frame_index"],
                        "to_frame": next_record["frame_index"],
                        "track_id": current_det["track_id"],
                        "class_id": current_det["class_id"],
                    }
                )
                continue

            next_index, next_det = best_match
            used_next_indices.add(next_index)
            event = {
                "from_frame": current_record["frame_index"],
                "to_frame": next_record["frame_index"],
                "class_id": current_det["class_id"],
                "iou": best_iou,
                "from_track_id": current_det["track_id"],
                "to_track_id": next_det["track_id"],
            }
            if current_det["track_id"] == next_det["track_id"]:
                kept_events.append(event)
            else:
                switched_events.append(event)
    return kept_events, switched_events, lost_events


def analyze_occlusion_window(frame_records, selected_frames, iou_threshold=0.5):
    selected_records = [record for record in frame_records if record["frame_index"] in selected_frames]
    selected_records.sort(key=lambda item: item["frame_index"])
    kept_events, switched_events, lost_events = _analyze_record_pairs(selected_records, iou_threshold=iou_threshold)

    return {
        "frames": [record["frame_index"] for record in selected_records],
        "transition_count": max(len(selected_records) - 1, 0),
        "kept_count": len(kept_events),
        "switch_count": len(switched_events),
        "lost_count": len(lost_events),
        "kept_events": kept_events,
        "switched_events": switched_events,
        "lost_events": lost_events,
    }


def analyze_tracking_transitions(frame_records, iou_threshold=0.5):
    records = sorted(frame_records, key=lambda item: item["frame_index"])
    kept_events, switched_events, lost_events = _analyze_record_pairs(records, iou_threshold=iou_threshold)
    return {
        "transition_count": max(len(records) - 1, 0),
        "kept_count": len(kept_events),
        "switch_count": len(switched_events),
        "lost_count": len(lost_events),
        "kept_events": kept_events,
        "switched_events": switched_events,
        "lost_events": lost_events,
    }
