import os
from typing import Dict, Any, List, Tuple
import json
import numpy as np


def load_shoe_labels(json_path: str) -> Dict[str, Any]:
    """
    Load shoe classification labels from a JSON file stored in assets/shoes.
    The expected format is a dict mapping tracker_id strings to records like:
        {
          "1": {
            "class": "Sneakers",
            "confidence": 0.659,
            "frame": 1313,
            "area": 3650.5,
            "crop_size": "64x64"
          }
    """
    data = {"labels": [], "meta": {"source": json_path}}
    if not os.path.exists(json_path):
        return data

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return data

    labels: List[Dict[str, Any]] = []
    for tracker_id_str, item in (raw or {}).items():
        if not isinstance(item, dict):
            continue

        cls = item.get("class")
        # Skip small boxes by requirement
        if str(cls).lower() == "small_bbox":
            continue

        try:
            frame = int(item.get("frame", -1))
            tracker_id = int(tracker_id_str)  # Convert key to int
        except (ValueError, TypeError):
            continue

        if frame < 0:
            continue

        conf = item.get("confidence", None)
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        rec = dict(item)
        rec["frame"] = frame
        rec["class"] = cls
        rec["confidence"] = conf
        rec["tracker_id"] = tracker_id  # Add tracker_id to the record

        labels.append(rec)

    data["labels"] = labels
    return data


def get_frame_shoes(data: Dict[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    """Return list of shoe label records for a given 0-based frame index.
    Each item includes at least: {"frame", "class", "confidence", "tracker_id"}
    """
    labels = data.get("labels", [])
    if not labels:
        return []
    out: List[Dict[str, Any]] = []
    for it in labels:
        try:
            if int(it.get("frame", -1)) == int(frame_idx):
                out.append(it)
        except Exception:
            continue
    return out

def get_tracker_shoes_static(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Return a dictionary {tracker_id -> shoe_info} without frame association.
    It is assumed that each record in the JSON describes shoes for the track as a whole,
    and the "frame" field is merely the frame where it was detected.
    If there are multiple records for one tracker_id, take the record with the highest confidence.
    """
    labels = data.get("labels", []) or []
    result: Dict[int, Dict[str, Any]] = {}

    for it in labels:
        tracker_id = it.get("tracker_id")
        if tracker_id is None:
            continue
        try:
            tracker_id = int(tracker_id)
        except (TypeError, ValueError):
            continue

        cls = it.get("class", "Unknown")
        conf = it.get("confidence", 0.0) or 0.0

        prev = result.get(tracker_id)
        if prev is None or conf > prev.get("confidence", 0.0):
            result[tracker_id] = {
                "class": cls,
                "confidence": conf,
                "frame": it.get("frame", -1),
            }

    return result

def summarize_frame_shoes(data: Dict[str, Any], frame_idx: int) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Summarize shoes on a frame: returns (counts_by_class, avg_conf_by_class).
    Small bboxes are already filtered at load time.
    """
    items = get_frame_shoes(data, frame_idx)
    counts: Dict[str, int] = {}
    confs: Dict[str, List[float]] = {}
    for it in items:
        cls = it.get("class") or "Unknown"
        counts[cls] = counts.get(cls, 0) + 1
        c = it.get("confidence")
        if c is not None:
            confs.setdefault(cls, []).append(float(c))
    avg_conf: Dict[str, float] = {}
    for cls, arr in confs.items():
        if arr:
            avg_conf[cls] = float(np.mean(arr))
    return counts, avg_conf


def summarize_all_shoes(data: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, float]]:

    labels = data.get("labels", []) or []
    counts: Dict[str, int] = {}
    confs: Dict[str, List[float]] = {}
    for it in labels:
        cls = it.get("class") or "Unknown"
        counts[cls] = counts.get(cls, 0) + 1
        c = it.get("confidence")
        if c is not None:
            confs.setdefault(cls, []).append(float(c))
    avg_conf: Dict[str, float] = {}
    for cls, arr in confs.items():
        if arr:
            avg_conf[cls] = float(np.mean(arr))
    return counts, avg_conf


def draw_shoes_summary_on_image(image_bgr, counts: Dict[str, int], avg_conf: Dict[str, float]):
    """
    Draw a compact legend with shoe classes, counts and avg confidence on the image.
    Returns a new image (BGR). If cv2 is not available, returns input image.
    """
    try:
        import cv2
    except Exception:
        return image_bgr

    if image_bgr is None:
        return None

    out = image_bgr.copy()

    # Prepare text lines
    lines: List[str] = []
    classes = sorted(counts.keys())
    for cls in classes:
        cnt = counts.get(cls, 0)
        conf = avg_conf.get(cls, None)
        if conf is not None:
            lines.append(f"{cls}: {cnt} (avg {conf:.2f})")
        else:
            lines.append(f"{cls}: {cnt}")

    if not lines:
        return out

    # Visual params
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    pad = 8
    line_gap = 6

    # Compute box size
    text_sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    box_w = max(w for (w, h) in text_sizes) + 2 * pad
    box_h = sum(h for (w, h) in text_sizes) + (len(lines) - 1) * line_gap + 2 * pad

    x1, y1 = 10, 10  # top-left
    x2, y2 = x1 + box_w, y1 + box_h

    # Background rectangle (semi-transparent dark)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), thickness=-1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # Title
    title = "Shoes"
    (tw, th), base = cv2.getTextSize(title, font, 0.7, 2)
    cv2.putText(out, title, (x1 + pad, y1 + pad + th), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Items
    cur_y = y1 + pad + th + 10
    for t, (w, h) in zip(lines, text_sizes):
        cv2.putText(out, t, (x1 + pad, cur_y + h), font, scale, (230, 230, 230), 2, cv2.LINE_AA)
        cur_y += h + line_gap

    return out