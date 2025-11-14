def get_video_info(video_path):
    """Получить информацию о видео. Работает даже без OpenCV.

    Если OpenCV (cv2) не установлен, возвращает нули, чтобы приложение не падало.
    """
    try:
        import cv2  # локальный импорт, чтобы не падать при отсутствии зависимости
    except Exception:
        return {'fps': 0, 'frame_count': 0, 'duration': 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'fps': 0, 'frame_count': 0, 'duration': 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps and fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': int(duration)
    }