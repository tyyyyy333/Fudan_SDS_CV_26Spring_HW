def _point_side(line, point):
    x1, y1, x2, y2 = line
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


class LineCounter:
    def __init__(self, line, count_once=True, allowed_class_ids=None):
        self.line = line
        self.count_once = count_once
        self.allowed_class_ids = allowed_class_ids
        self.total_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.last_side_by_track = {}
        self.counted_track_ids = set()
        self.counts_by_class = {}
        self.events = []

    def update(self, track_id, center, frame_index, class_id=None, time_seconds=None):
        if self.allowed_class_ids is not None and class_id not in self.allowed_class_ids:
            return None

        current_side = _point_side(self.line, center)
        previous_side = self.last_side_by_track.get(track_id)
        event = None
        already_counted = track_id in self.counted_track_ids

        if (
            previous_side is not None
            and previous_side != 0
            and current_side != 0
            and current_side != previous_side
            and (not self.count_once or not already_counted)
        ):
            self.total_count += 1
            if current_side > previous_side:
                self.positive_count += 1
                direction = "negative_to_positive"
            else:
                self.negative_count += 1
                direction = "positive_to_negative"

            if self.count_once:
                self.counted_track_ids.add(track_id)
            if class_id is not None:
                self.counts_by_class[class_id] = self.counts_by_class.get(class_id, 0) + 1

            event = {
                "track_id": track_id,
                "frame_index": frame_index,
                "center_x": center[0],
                "center_y": center[1],
                "direction": direction,
            }
            if class_id is not None:
                event["class_id"] = class_id
            if time_seconds is not None:
                event["time_seconds"] = time_seconds
            self.events.append(event)

        if current_side != 0:
            self.last_side_by_track[track_id] = current_side
        return event
