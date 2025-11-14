"""Utility package for the Streamlit Safe Play app.

This module exposes commonly used helpers directly from the
`utils` namespace for convenient imports in `app.py`, e.g.:

    from utils import get_video_info

It uses lazy imports so that heavy dependencies (like OpenCV)
are only imported when the corresponding function is actually
used.
"""

from typing import Any, Dict

__all__ = ["get_video_info"]


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Lazy wrapper for `video_processor.get_video_info`.

    This avoids importing heavy dependencies at package import time.
    """
    from .video_processor import get_video_info as _get_video_info

    return _get_video_info(video_path)
