import json
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def load_detections(json_path: str) -> Dict[str, Any]:
    """Load YOLO detections JSON.

    Expected schema contains keys: video_info, detection_info, results (list per frame).
    Returns the parsed dictionary.
    """
    if not os.path.exists(json_path):
        return {"video_info": {}, "detection_info": {}, "results": []}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # basic validation
        if not isinstance(data.get("results", []), list):
            data["results"] = []
        return data
    except Exception:
        return {"video_info": {}, "detection_info": {}, "results": []}


def get_frame_detections(
        data: Dict[str, Any],
        frame_idx: int,
        min_confidence: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Return detections list for a specific frame index.

    Args:
        data: Detection data dictionary
        frame_idx: Frame index
        min_confidence: Minimum confidence threshold (0.0-1.0). If None, no filtering.

    Returns:
        List of detections for the frame
    """
    results = data.get("results", [])
    detections = []

    # Many exporters save frames in order; we can index directly if frames are contiguous
    if 0 <= frame_idx < len(results):
        res = results[frame_idx]
        # ensure the frame number matches; fallback to search if mismatch
        if isinstance(res, dict) and res.get("frame") == frame_idx:
            detections = res.get("detections", []) or []
        else:
            # fallback: search by frame field
            for item in results:
                try:
                    if int(item.get("frame")) == int(frame_idx):
                        detections = item.get("detections", []) or []
                        break
                except Exception:
                    continue
    else:
        # fallback: search by frame field
        for item in results:
            try:
                if int(item.get("frame")) == int(frame_idx):
                    detections = item.get("detections", []) or []
                    break
            except Exception:
                continue

    # Apply confidence filtering if specified
    if min_confidence is not None and detections:
        detections = [
            det for det in detections
            if det.get("confidence", 0.0) >= min_confidence
        ]

    return detections


def compute_avg_detections(data: Dict[str, Any], min_confidence: Optional[float] = None) -> float:
    """Compute average number of detections per frame from JSON.

    Args:
        data: Detection data dictionary
        min_confidence: Minimum confidence threshold for filtering

    Returns:
        Average detections per frame
    """
    results = data.get("results", [])
    if not results:
        return 0.0
    total = 0
    count = 0
    for item in results:
        if isinstance(item, dict):
            dets = item.get("detections", [])
            if isinstance(dets, list):
                # Apply confidence filter if specified
                if min_confidence is not None:
                    dets = [d for d in dets if d.get("confidence", 0.0) >= min_confidence]
                total += len(dets)
                count += 1
    if count == 0:
        return 0.0
    return float(total) / float(count)


def muted_color_palette(n: int) -> List[Tuple[int, int, int]]:
    """Return n muted RGB colors.
    Uses a fixed pleasant palette cycled as needed.
    """
    base = [
        (108, 117, 125),  # gray-ish
        (52, 152, 219),  # muted blue
        (46, 204, 113),  # muted green
        (241, 196, 15),  # muted yellow
        (231, 76, 60),  # muted red
        (155, 89, 182),  # muted purple
        (26, 188, 156),  # muted teal
        (243, 156, 18),  # orange
        (149, 165, 166),  # concrete
        (52, 73, 94),  # wet asphalt
    ]
    if n <= len(base):
        return base[:n]
    # cycle
    colors = []
    for i in range(n):
        colors.append(base[i % len(base)])
    return colors


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a single frame (BGR) from video using OpenCV. Returns None on failure."""
    try:
        import cv2
    except Exception:
        return None
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    ok = False
    frame = None

    try:
        # Определяем реальное количество кадров и ограничиваем индекс
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            frame_idx = max(0, min(int(frame_idx), total_frames - 1))
        else:
            frame_idx = max(0, int(frame_idx))

        # Попытка прямого перехода к кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        # Если не удалось прочитать кадр (часто на самом конце видео),
        # попробуем отступить на пару кадров назад
        if not ok and total_frames > 0:
            fallback_idx = max(0, min(frame_idx - 2, total_frames - 1))
            if fallback_idx != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
                ok, frame = cap.read()

        # Дополнительный фолбэк: последовательное чтение с начала
        if not ok and total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            target_idx = max(0, min(frame_idx, total_frames - 1))
            cur_idx = 0
            while cur_idx <= target_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                if cur_idx == target_idx:
                    break
                cur_idx += 1
    finally:
        cap.release()

    if not ok:
        return None
    return frame  # BGR


def draw_bboxes_on_image(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes with muted colors and labels onto a BGR image.

    Each bbox in detections is expected to have keys: class, confidence, bbox{x1,y1,x2,y2}.
    Returns a new BGR image with overlays.
    """
    if image_bgr is None:
        return None
    try:
        import cv2
    except Exception:
        return image_bgr

    out = image_bgr.copy()
    colors = muted_color_palette(max(1, len(detections)))

    for i, det in enumerate(detections):
        bbox = (det or {}).get("bbox", {})
        try:
            x1 = int(round(bbox.get("x1", 0)))
            y1 = int(round(bbox.get("y1", 0)))
            x2 = int(round(bbox.get("x2", 0)))
            y2 = int(round(bbox.get("y2", 0)))
        except Exception:
            continue
        color = colors[i % len(colors)]
        # draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        # label
        cls_name = str(det.get("class", "obj"))
        conf = det.get("confidence", None)
        label = f"{cls_name}" if conf is None else f"conf: {conf:.2f}"
        # background for text
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        x_bg2 = x1 + tw + 6
        y_bg2 = y1 + th + 4
        # semi-transparent bg
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x_bg2, y_bg2), color, thickness=-1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        # text
        cv2.putText(out, label, (x1 + 3, y1 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    return out


def create_video_with_detections(
        video_path: str,
        det_data: Dict[str, Any],
        output_path: str,
        min_confidence: Optional[float] = None,
        progress_callback=None
) -> bool:
    """Create a video with detections drawn on each frame.

    Args:
        video_path: Path to input video
        det_data: Detection data dictionary
        output_path: Path to save output video
        min_confidence: Minimum confidence threshold
        progress_callback: Optional callback function for progress updates (receives frame_idx, total_frames)

    Returns:
        True if successful, False otherwise
    """
    try:
        import cv2
    except Exception:
        return False

    if not os.path.exists(video_path):
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for current frame
            dets = get_frame_detections(det_data, frame_idx, min_confidence)

            # Draw detections
            frame_with_dets = draw_bboxes_on_image(frame, dets)

            # Write frame
            out.write(frame_with_dets)

            # Progress callback
            if progress_callback:
                progress_callback(frame_idx, total_frames)

            frame_idx += 1

        return True

    except Exception:
        return False

    finally:
        cap.release()
        if 'out' in locals():
            out.release()