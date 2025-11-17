import os
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

import numpy as np


# ----------------------------
# Parsing & data access
# ----------------------------

def load_mot_tracks(txt_path: str) -> Dict[str, Any]:
    """
    Load OC-SORT (MOT-format) tracks from a txt file.

    Expected MOT format per line (at least the first 7 fields):
        frame, id, x, y, w, h, conf, ...
    - frame: 1-based frame index in common MOT exports; we'll convert to 0-based
    - id: track id (int)
    - x, y, w, h: bbox in pixels (top-left X/Y, width, height)
    - conf: optional confidence (float)

    Returns a dict with keys:
        tracks: List[Dict] with per-frame records
            Each item: {frame, id, bbox{x1,y1,x2,y2}, conf}
        meta: Dict with input file path
    """
    data = {"tracks": [], "meta": {"source": txt_path}}
    if not os.path.exists(txt_path):
        return data

    tracks: List[Dict[str, Any]] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    # Some files could be space-separated
                    parts = [p.strip() for p in line.split()]  # fallback
                if len(parts) < 6:
                    continue
                try:
                    frame = int(float(parts[0]))  # some exporters use float
                    track_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6]) if len(parts) >= 7 else None
                except Exception:
                    continue

                # Convert 1-based to 0-based if looks like 1-based
                # Heuristic: if frame > 0, subtract 1 to map to 0-based for internal indexing
                frame0 = max(0, frame - 1)

                x1 = int(round(x))
                y1 = int(round(y))
                x2 = int(round(x + w))
                y2 = int(round(y + h))

                tracks.append({
                    "frame": frame0,
                    "id": track_id,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "conf": conf,
                })
    except Exception:
        # On any parsing error, just return what we have so far
        pass

    data["tracks"] = tracks
    return data


def get_frame_tracks(data: Dict[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    """
    Return list of tracks for a given 0-based frame index.

    Each item: {id, bbox{x1,y1,x2,y2}, conf}
    """
    tracks = data.get("tracks", [])
    if not tracks:
        return []
    out = []
    for item in tracks:
        try:
            if int(item.get("frame", -1)) == int(frame_idx):
                out.append({
                    "id": int(item.get("id")),
                    "bbox": dict(item.get("bbox", {})),
                    "conf": item.get("conf"),
                })
        except Exception:
            continue
    return out


# ----------------------------
# Drawing helpers
# ----------------------------

def pleasant_palette() -> List[Tuple[int, int, int]]:
    """A fixed set of pleasant BGR colors (muted but distinct)."""
    # BGR tuples chosen to be eye-pleasing on light background
    return [
        (125, 117, 108),  # muted gray-brown
        (219, 152, 52),   # muted blue-ish (note: BGR order)
        (113, 204, 46),   # muted green
        (15, 196, 241),   # muted yellow-ish cyan
        (60, 76, 231),    # muted red-ish (in BGR becomes purple-ish)
        (182, 89, 155),   # muted purple
        (156, 188, 26),   # muted teal/green
        (18, 156, 243),   # orange-like in BGR
        (166, 165, 149),  # concrete
        (94, 73, 52),     # wet asphalt brownish
    ]


def id_color(track_id: int, palette: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[int, int, int]:
    """Deterministic color for a track id using palette cycling."""
    pal = palette or pleasant_palette()
    if not pal:
        return (80, 200, 120)  # fallback
    idx = abs(int(track_id)) % len(pal)
    return pal[idx]


def draw_tracks_on_image(
        image_bgr: np.ndarray,
        tracks: List[Dict[str, Any]],
        track_history: Optional[Dict[int, deque]] = None,
        smooth_trail: bool = True,
        smooth_iters: int = 2,
        trail_thickness: int = 2
) -> np.ndarray:
    """
    Draw track bboxes with consistent pleasant colors and ID labels above the box.

    Each item in `tracks` must have keys: id, bbox{x1,y1,x2,y2}

    Args:
        image_bgr: Input image (BGR)
        tracks: List of track dicts for the current frame
        track_history: Optional mapping track_id -> deque/list of (x, y) points for the trail
        smooth_trail: If True, applies Chaikin smoothing to trails for a smoother look
        smooth_iters: Number of Chaikin iterations (2–3 is usually enough)
        trail_thickness: Thickness of trail line
    """
    if image_bgr is None:
        return None
    try:
        import cv2
    except Exception:
        return image_bgr

    out = image_bgr.copy()

    for tr in tracks:
        bbox = (tr or {}).get("bbox", {})
        try:
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))
        except Exception:
            continue
        tid = tr.get("id", None)
        color = id_color(int(tid) if tid is not None else 0)

        # Draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

        # Label 'ID {id}' above the bbox
        label = f"ID {tid}" if tid is not None else "ID ?"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        th_tot = th + baseline
        pad = 4
        # background box above top-left corner
        x_bg1 = x1
        y_bg1 = max(0, y1 - th_tot - pad - 2)
        x_bg2 = x1 + tw + 2 * pad
        y_bg2 = y1 - 2

        # semi-transparent background for legibility
        overlay = out.copy()
        cv2.rectangle(overlay, (x_bg1, y_bg1), (x_bg2, y_bg2), color, thickness=-1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

        # text on top
        txt_org = (x_bg1 + pad, y_bg2 - pad)
        cv2.putText(out, label, txt_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 20, 60), 2, cv2.LINE_AA)

    # ----- DRAW TRAILS -----
    # Рисуем «хвосты» только если передана история
    if track_history is not None:
        def chaikin(points: List[Tuple[int, int]], iterations: int) -> List[Tuple[int, int]]:
            if len(points) < 3 or iterations <= 0:
                return list(points)
            pts = [np.array(p, dtype=np.float32) for p in points]
            for _ in range(iterations):
                if len(pts) < 3:
                    break
                new_pts = [pts[0]]
                for i in range(len(pts) - 1):
                    p = pts[i]
                    q = pts[i + 1]
                    Q = 0.75 * p + 0.25 * q
                    R = 0.25 * p + 0.75 * q
                    new_pts.extend([Q, R])
                new_pts.append(pts[-1])
                pts = new_pts
            return [(int(round(p[0])), int(round(p[1]))) for p in pts]

        for tr in tracks:
            tid = tr.get("id")
            if tid not in track_history:
                continue

            pts = list(track_history[tid])
            if len(pts) < 2:
                continue

            color = id_color(int(tid))

            # Optionally smooth the trail
            draw_pts = chaikin(pts, smooth_iters) if smooth_trail else pts

            # Draw antialiased polyline
            pts_np = np.array(draw_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts_np], isClosed=False, color=color, thickness=trail_thickness, lineType=cv2.LINE_AA)

    return out



def create_video_with_tracks(
        video_path: str,
        tracks_data: Dict[str, Any],
        output_path: str,
        progress_callback=None
) -> bool:
    """Create a video with OC-SORT tracks drawn on each frame.

    Args:
        video_path: Path to input video.
        tracks_data: Parsed MOT-format tracks as returned by `load_mot_tracks`.
        output_path: Path to save the output video (e.g., .mp4).
        progress_callback: Optional function(frame_idx: int, total_frames: int) -> None for UI progress.

    Returns:
        True on success, False otherwise.
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        track_history = {}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = get_frame_tracks(tracks_data or {}, frame_idx)
            # --- обновление истории ---
            for tr in tracks:
                tid = tr["id"]
                bbox = tr["bbox"]
                cx = int((bbox["x1"] + bbox["x2"]) / 2)
                cy = int(bbox["y2"])  # нижняя граница

                if tid not in track_history:
                    track_history[tid] = deque(maxlen=25)
                track_history[tid].append((cx, cy))

            # --- передача истории в рисовалку ---
            frame_with_tracks = draw_tracks_on_image(frame, tracks, track_history)

            out.write(frame_with_tracks)

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
