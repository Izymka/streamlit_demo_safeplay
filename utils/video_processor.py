def get_video_info(video_path):
    """Получить информацию о видео. Работает даже без OpenCV.

    Если OpenCV (cv2) не установлен, возвращает нули, чтобы приложение не падало.
    """
    try:
        import cv2  # локальный импорт, чтобы не падать при отсутствии зависимости
        import os
    except Exception:
        return {
            'fps': 0,
            'frame_count': 0,
            'duration': 0,
            'width': 0,
            'height': 0,
            'bitrate': 0,
            'codec': 'N/A',
            'container': 'N/A'
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'fps': 0,
            'frame_count': 0,
            'duration': 0,
            'width': 0,
            'height': 0,
            'bitrate': 0,
            'codec': 'N/A',
            'container': 'N/A'
        }

    # Базовая информация
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps and fps > 0 else 0
    
    # Разрешение
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    # Попытка получить информацию о кодеке
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if not codec or codec == '\x00\x00\x00\x00':
        codec = 'N/A'
    
    cap.release()
    
    # Формат контейнера из расширения файла
    container = os.path.splitext(video_path)[1][1:].upper() if os.path.exists(video_path) else 'N/A'
    
    # Приблизительный битрейт (размер файла / длительность)
    bitrate = 0
    try:
        if os.path.exists(video_path) and duration > 0:
            file_size_bits = os.path.getsize(video_path) * 8
            bitrate = int(file_size_bits / duration)  # bits per second
    except Exception:
        bitrate = 0
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': int(duration),
        'width': width,
        'height': height,
        'bitrate': bitrate,
        'codec': codec,
        'container': container
    }