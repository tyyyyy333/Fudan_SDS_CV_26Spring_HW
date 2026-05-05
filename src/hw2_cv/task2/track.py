import subprocess
from pathlib import Path

import cv2
from tqdm import tqdm

from hw2_cv.task2.analysis import analyze_occlusion_window, analyze_tracking_transitions, summarize_tracks
from hw2_cv.task2.line_counter import LineCounter
from hw2_cv.task2.visualize import draw_tracks
from hw2_cv.utils import ensure_dir, log_info, prepare_run, save_json, save_jsonl


def _manual_selected_frames(analysis_cfg):
    occlusion_cfg = analysis_cfg.get("occlusion", {})
    if not occlusion_cfg:
        return set()
    explicit_indices = occlusion_cfg.get("indices")
    if explicit_indices:
        return {int(index) for index in explicit_indices}
    if "start_frame" not in occlusion_cfg:
        return set()
    start_frame = int(occlusion_cfg.get("start_frame", 0))
    num_frames = int(occlusion_cfg.get("num_frames", 4))
    return set(range(start_frame, start_frame + num_frames))


def _window_indices(center_frame, radius, total_frames):
    start_frame = max(int(center_frame) - int(radius), 0)
    end_frame = min(int(center_frame) + int(radius), max(total_frames - 1, 0))
    return list(range(start_frame, end_frame + 1))


def _pick_interesting_frames(frame_records, transition_summary, crossing_events, occlusion_cfg, total_frames):
    manual_frames = _manual_selected_frames({"occlusion": occlusion_cfg})
    if manual_frames:
        return manual_frames, "manual"

    radius = int(occlusion_cfg.get("event_radius", 2))
    max_windows = int(occlusion_cfg.get("max_windows", 2))
    selected_frames = []
    selected_set = set()

    priority_events = transition_summary["switched_events"] + transition_summary["lost_events"]
    for event in priority_events[:max_windows]:
        center_frame = event["to_frame"] if "to_frame" in event else event["from_frame"]
        for frame_index in _window_indices(center_frame, radius, total_frames):
            if frame_index not in selected_set:
                selected_frames.append(frame_index)
                selected_set.add(frame_index)
    if selected_set:
        return selected_set, "tracking_transition"

    for event in crossing_events[:max_windows]:
        for frame_index in _window_indices(event["frame_index"], radius, total_frames):
            if frame_index not in selected_set:
                selected_frames.append(frame_index)
                selected_set.add(frame_index)
    if selected_set:
        return selected_set, "crossing_event"

    return set(), "empty"


def _export_video_frames(video_path, selected_frames, output_dir):
    selected_frames = sorted(set(int(frame_index) for frame_index in selected_frames))
    if not selected_frames:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video for frame export: {video_path}")

    saved_paths = []
    try:
        next_index = 0
        frame_index = 0
        while next_index < len(selected_frames):
            success, frame = capture.read()
            if not success or frame is None:
                break
            target_frame = selected_frames[next_index]
            if frame_index < target_frame:
                frame_index += 1
                continue
            if frame_index == target_frame:
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(str(frame_path))
                next_index += 1
            frame_index += 1
    finally:
        capture.release()
    return saved_paths


def _clear_exported_jpegs(directory):
    for image_path in directory.glob("*.jpg"):
        image_path.unlink()


def _transcode_mp4(temp_video_path, output_path):
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("imageio-ffmpeg is required for mp4 export in this environment.") from error

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(temp_video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcoding failed for {output_path}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _video_writer(output_path, fps, width, height):
    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer_path = output_path
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer_path = output_path.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(str(writer_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {writer_path}")
    return writer, Path(writer_path)


def _video_meta(video_path):
    capture = cv2.VideoCapture(video_path)
    success, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if not success or frame is None:
        raise FileNotFoundError(f"Unable to read video: {video_path}")
    height, width = frame.shape[:2]
    return fps, width, height, total_frames


def run_tracking(config):
    from ultralytics import YOLO

    config, output_dir, _ = prepare_run(config)
    track_cfg = config["track"]
    counting_cfg = track_cfg.get("counting", {})
    analysis_cfg = track_cfg.get("analysis", {})
    export_cfg = track_cfg.get("export", {})

    save_dir = ensure_dir(track_cfg.get("save_dir", output_dir / "track"))
    sampled_frame_dir = ensure_dir(save_dir / export_cfg.get("occlusion_dir_name", "occlusion_frames"))
    event_frame_dir = ensure_dir(save_dir / export_cfg.get("event_dir_name", "crossing_frames"))
    _clear_exported_jpegs(sampled_frame_dir)
    _clear_exported_jpegs(event_frame_dir)
    frame_records_path = save_dir / export_cfg.get("frame_records_filename", "frame_records.jsonl")
    events_path = save_dir / export_cfg.get("events_filename", "crossing_events.json")
    track_summary_path = save_dir / export_cfg.get("track_summary_filename", "track_summary.json")
    occlusion_summary_path = save_dir / export_cfg.get("occlusion_summary_filename", "occlusion_analysis.json")

    video_path = track_cfg["video"]
    fps, width, height, total_frames = _video_meta(video_path)
    line = tuple(track_cfg["line"])
    allowed_classes = counting_cfg.get("allowed_class_ids")
    line_counter = LineCounter(
        line=line,
        count_once=bool(counting_cfg.get("count_once", True)),
        allowed_class_ids=None if allowed_classes is None else {int(class_id) for class_id in allowed_classes},
    )
    output_video_path = save_dir / export_cfg.get("tracked_video_filename", "tracked.mp4")

    model = YOLO(track_cfg["weights"])
    frame_records = []
    event_frames = []
    track_states = {}

    writer, writer_path = _video_writer(output_video_path, fps, width, height)

    tracker_name = track_cfg.get("tracker", "botsort.yaml")
    classes = track_cfg.get("classes")
    max_det = int(track_cfg.get("max_det", 300))
    frame_index = 0
    log_info(
        f"[task2-track] video={video_path} | tracker={tracker_name} | conf={track_cfg.get('conf', 0.25)} | "
        f"iou={track_cfg.get('iou', 0.5)} | save_dir={save_dir}"
    )

    try:
        stream = model.track(
            source=video_path,
            conf=float(track_cfg.get("conf", 0.25)),
            iou=float(track_cfg.get("iou", 0.5)),
            tracker=tracker_name,
            classes=classes,
            max_det=max_det,
            persist=True,
            stream=True,
            verbose=False,
        )
        progress = tqdm(stream, total=total_frames or None, desc="task2 track", leave=False)
        for result in progress:
            frame = result.orig_img.copy()
            detections = []
            frame_events = []
            boxes = []
            class_ids = []
            confidences = []
            track_ids = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.float().cpu().tolist()
                if result.boxes.id is None:
                    track_ids = [None] * len(boxes)
                else:
                    track_ids = result.boxes.id.int().cpu().tolist()

                for box, class_id, confidence, track_id in zip(boxes, class_ids, confidences, track_ids):
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    time_seconds = frame_index / fps if fps > 0 else None
                    detections.append(
                        {
                            "track_id": track_id,
                            "class_id": class_id,
                            "confidence": float(confidence),
                            "box": box,
                            "center": [center_x, center_y],
                        }
                    )

                    if track_id is not None:
                        state = track_states.setdefault(
                            track_id,
                            {
                                "first_frame": frame_index,
                                "last_frame": frame_index,
                                "frames_seen": 0,
                                "class_counts": {},
                            },
                        )
                        state["last_frame"] = frame_index
                        state["frames_seen"] += 1
                        state["class_counts"][class_id] = state["class_counts"].get(class_id, 0) + 1
                        event = line_counter.update(
                            track_id=track_id,
                            center=(center_x, center_y),
                            frame_index=frame_index,
                            class_id=class_id,
                            time_seconds=time_seconds,
                        )
                        if event is not None:
                            frame_events.append(event)

            annotated = draw_tracks(
                frame=frame,
                boxes=boxes,
                track_ids=track_ids,
                class_ids=class_ids,
                names=result.names,
                line=line,
                total_count=line_counter.total_count,
                positive_count=line_counter.positive_count,
                negative_count=line_counter.negative_count,
            )
            writer.write(annotated)

            for event in frame_events:
                frame_path = event_frame_dir / (
                    f"frame_{frame_index:06d}_track_{event['track_id']}_{event['direction']}.jpg"
                )
                cv2.imwrite(str(frame_path), annotated)
                event_with_frame = dict(event)
                event_with_frame["frame_path"] = str(frame_path)
                event_frames.append(event_with_frame)

            frame_records.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps if fps > 0 else None,
                    "detections": detections,
                }
            )
            frame_index += 1
            progress.set_postfix(count=line_counter.total_count, tracks=len(track_states))
    finally:
        writer.release()

    if output_video_path.suffix.lower() == ".mp4":
        _transcode_mp4(writer_path, output_video_path)
        writer_path.unlink(missing_ok=True)

    track_summary = summarize_tracks(track_states)
    transition_summary = analyze_tracking_transitions(
        frame_records=frame_records,
        iou_threshold=float(analysis_cfg.get("match_iou_threshold", 0.5)),
    )
    selected_frames, selection_strategy = _pick_interesting_frames(
        frame_records=frame_records,
        transition_summary=transition_summary,
        crossing_events=line_counter.events,
        occlusion_cfg=analysis_cfg.get("occlusion", {}),
        total_frames=frame_index,
    )
    sampled_frames = _export_video_frames(
        video_path=output_video_path,
        selected_frames=selected_frames,
        output_dir=sampled_frame_dir,
    )
    occlusion_summary = analyze_occlusion_window(
        frame_records=frame_records,
        selected_frames=selected_frames,
        iou_threshold=float(analysis_cfg.get("match_iou_threshold", 0.5)),
    )
    occlusion_summary["selection_strategy"] = selection_strategy
    occlusion_summary["global_transition_summary"] = transition_summary
    summary = {
        "output_dir": str(output_dir),
        "tracked_video": str(output_video_path),
        "sampled_frames_dir": str(sampled_frame_dir),
        "sampled_frames": sampled_frames,
        "selected_frame_indices": sorted(selected_frames),
        "selected_frame_strategy": selection_strategy,
        "event_frames_dir": str(event_frame_dir),
        "event_frames": event_frames,
        "line_count": line_counter.total_count,
        "positive_count": line_counter.positive_count,
        "negative_count": line_counter.negative_count,
        "counts_by_class": dict(sorted(line_counter.counts_by_class.items())),
        "tracker": tracker_name,
        "events_path": str(events_path),
        "frame_records_path": str(frame_records_path),
        "track_summary_path": str(track_summary_path),
        "occlusion_summary_path": str(occlusion_summary_path),
        "processed_frames": frame_index,
    }

    save_json(line_counter.events, events_path)
    save_jsonl(frame_records, frame_records_path)
    save_json(track_summary, track_summary_path)
    save_json(occlusion_summary, occlusion_summary_path)
    save_json(summary, save_dir / "summary.json")
    log_info(
        f"[task2-track] done | frames={frame_index} | line_count={line_counter.total_count} | "
        f"events={len(line_counter.events)}"
    )
    return summary
