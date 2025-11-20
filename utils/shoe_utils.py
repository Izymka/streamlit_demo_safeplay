import os
from typing import Dict, Any, List, Tuple
import json
import numpy as np
from .text_utils import draw_text, text_size



def load_shoe_labels(json_path: str) -> Dict[str, Any]:
    """
    Load shoe classification labels from a JSON file stored in assets/shoes.
    The expected format is a dict mapping tracker_id strings to records like:
        {
          "1": {
            "class": "кроссовки",
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


def draw_shoes_summary_on_image(image_bgr, counts, avg_conf):
    import cv2
    from .text_utils import draw_text, text_size

    if image_bgr is None:
        return None

    out = image_bgr.copy()

    # Prepare lines
    lines = []
    for cls in sorted(counts.keys()):
        cnt = counts[cls]
        conf = avg_conf.get(cls)
        if conf is not None:
            lines.append(f"{cls}: {cnt} (avg {conf:.2f})")
        else:
            lines.append(f"{cls}: {cnt}")

    if not lines:
        return out

    pad = 8
    gap = 6

    # Заголовок
    title = "Обувь"   # теперь можно писать по-русски
    title_w, title_h = text_size(title, font_height=24)

    # Размеры блока
    text_sizes = [text_size(t, font_height=20) for t in lines]
    body_h = sum(h for _, h in text_sizes) + (len(lines)-1)*gap
    body_w = max(w for w,_ in text_sizes)

    box_w = max(title_w, body_w) + pad*2
    box_h = title_h + 10 + body_h + pad*2

    x1, y1 = 10, 10
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40,40,40), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # Draw title
    draw_text(out, title, (x1 + pad, y1 + pad + title_h), font_height=24)

    # Draw items
    cur_y = y1 + pad + title_h + 10
    for t in lines:
        w, h = text_size(t, font_height=20)
        draw_text(out, t, (x1 + pad, cur_y + h), font_height=20)
        cur_y += h + gap

    return out
