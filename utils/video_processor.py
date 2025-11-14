def get_video_info(video_path):
    """Получить информацию о видео максимально надёжно с несколькими резервами.

    Порядок попыток:
    1) OpenCV (быстро)
    2) ffprobe (если установлен ffmpeg/ffprobe в системе)

    Возвращает нули/"N/A", если ничего не удалось.
    """
    import os

    def _default():
        return {
            'fps': 0.0,
            'frame_count': 0,
            'duration': 0,
            'width': 0,
            'height': 0,
            'bitrate': 0,
            'codec': 'N/A',
            'container': 'N/A'
        }

    if not os.path.exists(video_path):
        return _default()

    info = _default()

    # 1) Попытка через OpenCV
    try:
        import cv2  # локальный импорт
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            # Кодек (FOURCC)
            try:
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
                if not codec or codec == '\x00\x00\x00\x00':
                    codec = 'N/A'
            except Exception:
                codec = 'N/A'

            cap.release()

            duration = 0
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps

            info.update({
                'fps': fps,
                'frame_count': frame_count,
                'duration': int(duration),
                'width': width,
                'height': height,
                'codec': codec,
                'container': os.path.splitext(video_path)[1][1:].upper() or 'N/A'
            })
        else:
            cap.release()
    except Exception:
        pass

    # 2) Резерв через ffprobe, если не хватает данных от OpenCV
    needs_ffprobe = (
        info['width'] == 0 or info['height'] == 0 or info['fps'] == 0 or info['duration'] == 0 or info['codec'] == 'N/A'
    )
    if needs_ffprobe:
        try:
            import json
            import subprocess
            cmd = [
                'ffprobe', '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,codec_name,avg_frame_rate,nb_read_frames,nb_frames',
                '-show_entries', 'format=duration,bit_rate',
                '-of', 'json',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout or '{}')

            # Стрим (видео)
            streams = data.get('streams', [])
            if streams:
                s0 = streams[0]
                width = int(s0.get('width') or 0)
                height = int(s0.get('height') or 0)
                codec_name = s0.get('codec_name') or 'N/A'
                avg_frame_rate = s0.get('avg_frame_rate') or '0/1'
                # fps из дроби
                try:
                    num, den = avg_frame_rate.split('/')
                    num = float(num)
                    den = float(den) if float(den) != 0 else 1.0
                    fps_ff = num / den if den != 0 else 0.0
                except Exception:
                    fps_ff = 0.0
                # количество кадров
                def _parse_int(x):
                    try:
                        return int(x)
                    except Exception:
                        try:
                            return int(float(x))
                        except Exception:
                            return 0
                nb_read_frames = _parse_int(s0.get('nb_read_frames')) if s0.get('nb_read_frames') is not None else 0
                nb_frames = _parse_int(s0.get('nb_frames')) if s0.get('nb_frames') is not None else 0

                if info['width'] == 0:
                    info['width'] = width
                if info['height'] == 0:
                    info['height'] = height
                if info['codec'] == 'N/A' and codec_name:
                    info['codec'] = codec_name.upper()
                if info['fps'] == 0 and fps_ff > 0:
                    info['fps'] = fps_ff
                if info['frame_count'] == 0:
                    if nb_read_frames > 0:
                        info['frame_count'] = nb_read_frames
                    elif nb_frames > 0:
                        info['frame_count'] = nb_frames

            # Формат
            fmt = data.get('format', {})
            try:
                duration_ff = float(fmt.get('duration') or 0.0)
            except Exception:
                duration_ff = 0.0
            if info['duration'] == 0 and duration_ff > 0:
                info['duration'] = int(duration_ff)
            # если кадров нет, но есть fps и длительность — оценим
            if info['frame_count'] == 0 and info['fps'] > 0 and duration_ff > 0:
                try:
                    info['frame_count'] = int(round(info['fps'] * duration_ff))
                except Exception:
                    pass

            # Контейнер
            if info['container'] == 'N/A':
                ext = os.path.splitext(video_path)[1][1:].upper()
                info['container'] = ext or 'N/A'
            # Битрейт из ffprobe, если доступен
            try:
                bit_rate_ff = int(fmt.get('bit_rate')) if fmt.get('bit_rate') else 0
                if info['bitrate'] == 0 and bit_rate_ff > 0:
                    info['bitrate'] = bit_rate_ff
            except Exception:
                pass
        except Exception:
            pass

    # Битрейт считаем по размеру файла и длительности, если она > 0
    try:
        if info['duration'] > 0:
            file_size_bits = os.path.getsize(video_path) * 8
            info['bitrate'] = int(file_size_bits / max(info['duration'], 1))
    except Exception:
        pass

    return info