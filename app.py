import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime

# utils
from utils.yolo_utils import (
    load_detections,
    get_frame_detections,
    compute_avg_detections,
    read_frame,
    draw_bboxes_on_image,
    create_video_with_detections,
)
from utils.track_utils import (
    load_mot_tracks,
    get_frame_tracks,
    draw_tracks_on_image,
    create_video_with_tracks,
)
from utils.shoe_utils import (
    load_shoe_labels,
    summarize_frame_shoes,
    draw_shoes_summary_on_image,
)
from utils.mask_utils import (
    load_mask,
    apply_mask_to_frame,
    get_masks_config
)

# Защитная обертка для получения инфо о видео без жесткой зависимости от cv2
def _is_cv2_usable():
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        _ = getattr(cv2, "__version__", None)
        return True
    except Exception:
        return False


_CV2_OK = _is_cv2_usable()


def get_video_info_safe(path: str) -> dict:
    """
    Пытается получить метаданные видео через utils.video_processor.get_video_info.
    Если cv2 недоступен/неработоспособен, возвращает пустой словарь без выброса исключений.
    """
    if not _CV2_OK:
        return {}
    try:
        from utils.video_processor import get_video_info
        return get_video_info(path)
    except Exception:
        return {}

def handle_video_mode_change():
    """Отключает все трекеры при переходе в режим видео"""
    st.session_state.track_id = False
    st.session_state.bot_sort = False
    st.session_state.bot_sort_reid = False
    st.session_state.byte_track = False

def select_tracker(tracker_name: str):
    """Активирует только выбранный трекер"""
    tracker_keys = ["track_id", "bot_sort", "bot_sort_reid", "byte_track"]

    for key in tracker_keys:
        st.session_state[key] = (key == tracker_name)

    st.session_state.video_mode = False

def handle_other_checkboxes_change():
    """Обработчик изменения других чекбоксов"""
    # Если любой другой чекбокс изменился и режим видео был включен - выключаем его
    if st.session_state.video_mode:
        st.session_state.video_mode = False


def handle_tracker_change():
    """Обработчик для трекер-чекбоксов: выключает режим видео и обеспечивает эксклюзивный выбор трекера"""
    # Всегда выключаем видеорежим, как и для других чекбоксов
    if st.session_state.video_mode:
        st.session_state.video_mode = False
    # Обеспечим, чтобы был активен только один трекер
    flags = [
        ("track_id", bool(st.session_state.get("track_id", False))),
        ("bot_sort", bool(st.session_state.get("bot_sort", False))),
        ("bot_sort_reid", bool(st.session_state.get("bot_sort_reid", False))),
        ("byte_track", bool(st.session_state.get("byte_track", False))),
    ]
    # Оставим включенным первый найденный True по приоритету, остальные выключим
    active_found = False
    for key, val in flags:
        if val and not active_found:
            active_found = True
        else:
            st.session_state[key] = False

@st.cache_data(show_spinner=False)
def _load_masks():
    """Загружает маски из assets/mask"""
    masks_config = get_masks_config()
    masks = {}
    for mask_name, config in masks_config.items():
        masks[mask_name] = load_mask(config["path"])
    return masks

# Загружаем маски
masks = _load_masks()
masks_config = get_masks_config()

# Инициализация состояния приложения
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0
if 'min_confidence' not in st.session_state:
    st.session_state.min_confidence = 0.0
if 'video_mode' not in st.session_state:
    st.session_state.video_mode = False
# Инициализация состояний для чекбоксов
if 'yolo_enabled' not in st.session_state:
    st.session_state.yolo_enabled = True
if 'track_id' not in st.session_state:
    st.session_state.track_id = False
if 'bot_sort' not in st.session_state:
    st.session_state.bot_sort = False
if 'bot_sort_reid' not in st.session_state:
    st.session_state.bot_sort_reid = False
if 'byte_track' not in st.session_state:
    st.session_state.byte_track = False
if 'shoe1' not in st.session_state:
    st.session_state.shoe1 = True
if 'floor' not in st.session_state:
    st.session_state.floor = True
if 'window' not in st.session_state:
    st.session_state.window = True

# Настройка страницы
st.set_page_config(
    page_title="Safe Play",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #e9ecef;
        color: #212529;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stat-label {
        color: #6c757d;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        color: #212529;
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("Safe Play")


# Создаем три колонки
col1, col2, col3 = st.columns([1, 3, 1.2])

# Левая панель - Возможности
with col1:
    st.markdown("### Возможности")

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    yolo_enabled = st.checkbox(
        "Детекции YOLO",
        value=st.session_state.yolo_enabled,
        key="yolo_enabled",
        on_change=handle_other_checkboxes_change
    )

    # Фильтр по уверенности
    if yolo_enabled:
        #st.markdown("**Фильтр уверенности**")
        st.session_state.min_confidence = st.slider(
            "Минимальная уверенность",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.min_confidence,
            step=0.05,
            format="%.2f"
        )

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    track_id = st.checkbox(
        "OC SORT",
        key="track_id",
        on_change=select_tracker,
        args=("track_id",),
    )

    #bot_sort = st.checkbox(
    #    "BoT SORT",
    #    key="bot_sort",
    #    on_change=select_tracker,
    #    args=("bot_sort",),
    #)

    bot_sort_reid = st.checkbox(
        "BoT SORT ReID",
        key="bot_sort_reid",
        on_change=select_tracker,
        args=("bot_sort_reid",),
    )

    #byte_track = st.checkbox(
    #    "BYTETrack",
    #    key="byte_track",
    #    on_change=select_tracker,
    #    args=("byte_track",),
    #)

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    shoe_classification_1 = st.checkbox(
        "Классификация обуви",
        value=st.session_state.shoe1,
        key="shoe1",
        on_change=handle_other_checkboxes_change
    )

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    # Режим видео
    st.markdown("**Режим воспроизведения**")
    video_mode = st.checkbox(
        "Видео",
        value=st.session_state.video_mode,
        key="video_mode",
        on_change=handle_video_mode_change
    )

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    st.markdown("**ROI Zone**")

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    floor = st.checkbox(
        "Пол",
        value=st.session_state.floor,
        key="floor",
        on_change=handle_other_checkboxes_change
    )

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    window = st.checkbox(
        "Окна",
        value=st.session_state.window,
        key="window",
        on_change=handle_other_checkboxes_change
    )

# Центральная панель - Видео
with col2:
    # Пути к данным
    video_file_path = "data/raw/basketball_000.mp4"
    det_json_path = "assets/yolo_det/basketball_000.json"

    # Определение активного трекера и файла разметки
    active_tracker_key = None
    active_tracker_label = None
    if st.session_state.get("track_id", False):
        active_tracker_key = "oc_sort"
        active_tracker_label = "OC SORT"
    elif st.session_state.get("bot_sort", False):
        active_tracker_key = "bot_sort"
        active_tracker_label = "Bot Sort"
    elif st.session_state.get("bot_sort_reid", False):
        active_tracker_key = "bot_sort_reid"
        active_tracker_label = "Bot Sort (ReID)"
    elif st.session_state.get("byte_track", False):
        active_tracker_key = "byte_track"
        active_tracker_label = "ByteTrack"

    # Выбор файла треков по активному трекеру
    if active_tracker_key == "oc_sort":
        tracks_txt_path = "assets/tracks/oc_sort_basketball_000.txt"
    elif active_tracker_key == "bot_sort":
        tracks_txt_path = "assets/tracks/bot_sort_basketball_000.txt"
    elif active_tracker_key == "bot_sort_reid":
        tracks_txt_path = "assets/tracks/bot_sort_reid_basketball_000.txt"
    elif active_tracker_key == "byte_track":
        tracks_txt_path = "assets/tracks/byte_track_basketball_000.txt"
    else:
        tracks_txt_path = None

    # Информация о видео
    frames = 0
    fps = 0.0
    if os.path.exists(video_file_path):
        try:
            vid_info = get_video_info_safe(video_file_path)
            st.session_state.video_duration = int(vid_info.get('duration', 0))
            frames = int(vid_info.get('frame_count', 0))
            fps = float(vid_info.get('fps', 0) or 0.0)
        except Exception:
            st.session_state.video_duration = 0
            frames = 0
            fps = 0.0
    else:
        st.session_state.video_duration = 0


    # Загрузка детекций
    @st.cache_data(show_spinner=False)
    def _load_json(path):
        return load_detections(path)

    @st.cache_data(show_spinner=False)
    def _load_tracks(path):
        return load_mot_tracks(path)

    @st.cache_data(show_spinner=False)
    def _load_shoes(path):
        return load_shoe_labels(path)

    det_data = _load_json(det_json_path) if os.path.exists(det_json_path) else {"results": []}
    tracks_data = _load_tracks(tracks_txt_path) if (tracks_txt_path and os.path.exists(tracks_txt_path)) else {"tracks": []}
    shoes_json_path = "assets/shoes/basketball_000.shoe_labels.json"
    shoes_data = _load_shoes(shoes_json_path) if os.path.exists(shoes_json_path) else {"labels": []}

    # Если число кадров неизвестно, попробуем взять из JSON
    if frames == 0:
        try:
            frames = int(det_data.get("video_info", {}).get("total_frames") or len(det_data.get("results", [])) or 0)
        except Exception:
            frames = len(det_data.get("results", []))

    # Кнопки навигации и управления
    control_cols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 4])

    with control_cols[0]:
        if st.button("⏮️ Начало"):
            st.session_state.current_frame = 0
            st.rerun()

    with control_cols[1]:
        if st.button("◀️ -10"):
            st.session_state.current_frame = max(0, st.session_state.current_frame - 10)
            st.rerun()

    with control_cols[2]:
        if st.button("◀️ -1"):
            st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
            st.rerun()

    with control_cols[3]:
        if st.button("▶️ +1"):
            max_frame_idx = max(0, (frames - 1) if frames else 0)
            st.session_state.current_frame = min(max_frame_idx, st.session_state.current_frame + 1)
            st.rerun()

    with control_cols[4]:
        if st.button("⏭️ +10"):
            max_frame_idx = max(0, (frames - 1) if frames else 0)
            st.session_state.current_frame = min(max_frame_idx, st.session_state.current_frame + 10)
            st.rerun()

    # Селектор кадра (если не в режиме видео)
    if not video_mode:
        max_frame_idx = max(0, (frames - 1) if frames else 0)
        st.session_state.current_frame = st.slider(
            "Кадр",
            min_value=0,
            max_value=max_frame_idx,
            value=int(st.session_state.get("current_frame", 0)),
            step=1,
        )

    # Отрисовка
    if os.path.exists(video_file_path):
        if video_mode:
            # Режим видео с детекциями
            st.markdown("#### 🎬 Режим видео")

            # Создаем колонки для кнопок
            col1, col2 = st.columns(2)

            with col1:
                # Кнопка для создания видео с детекциями
                if st.button("🎥 Создать видео с детекциями", use_container_width=True):
                    with st.spinner("Создание видео... Это может занять некоторое время."):
                        # Создаем временный файл
                        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        output_path = temp_output.name
                        temp_output.close()

                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()


                        def progress_callback(frame_idx, total_frames):
                            if total_frames > 0:
                                progress = frame_idx / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Обработка кадра {frame_idx}/{total_frames}")


                        # Создаем видео
                        success = create_video_with_detections(
                            video_file_path,
                            det_data,
                            output_path,
                            min_confidence=st.session_state.min_confidence,
                            progress_callback=progress_callback
                        )

                        if success:
                            st.success("✅ Видео успешно создано!")
                            # Показываем видео
                            #with open(output_path, "rb") as vf:
                            #    st.video(vf.read())

                            # Кнопка для скачивания
                            with open(output_path, "rb") as vf:
                                st.download_button(
                                    label="📥 Скачать видео",
                                    data=vf.read(),
                                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                    mime="video/mp4"
                                )

                            # Удаляем временный файл
                            try:
                                os.unlink(output_path)
                            except:
                                pass
                        else:
                            st.error("❌ Ошибка при создании видео")

                        progress_bar.empty()
                        status_text.empty()

            with col2:
                # Флаг: включать ли обувь в видео трекера
                include_shoes_in_tracker_video = st.checkbox("👟 Включить обувь в видео трекера", value=False)

                # Кнопка для создания видео с трекером (OC-SORT MOT)
                if st.button("🎥 Создать видео с трекером", use_container_width=True):
                    with st.spinner("Создание видео с трекером... Это может занять некоторое время."):
                        temp_output_tr = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        output_path_tr = temp_output_tr.name
                        temp_output_tr.close()

                        progress_bar_tr = st.progress(0)
                        status_text_tr = st.empty()


                        def progress_callback_tr(frame_idx, total_frames):
                            if total_frames > 0:
                                progress = frame_idx / total_frames
                                progress_bar_tr.progress(progress)
                                status_text_tr.text(f"Обработка кадра {frame_idx}/{total_frames}")


                        success_tr = create_video_with_tracks(
                            video_file_path,
                            tracks_data,
                            output_path_tr,
                            progress_callback=progress_callback_tr,
                            shoe_data=shoes_data if include_shoes_in_tracker_video else None,
                        )

                        if success_tr:
                            st.success("✅ Видео с трекером успешно создано!")
                            #with open(output_path_tr, "rb") as vf:
                            #    st.video(vf.read())
                            with open(output_path_tr, "rb") as vf:
                                st.download_button(
                                    label="📥 Скачать видео",
                                    data=vf.read(),
                                    file_name=f"tracks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                    mime="video/mp4"
                                )
                            try:
                                os.unlink(output_path_tr)
                            except:
                                pass
                        else:
                            st.error("❌ Ошибка при создании видео с трекером")

                        progress_bar_tr.empty()
                        status_text_tr.empty()

            # Показываем оригинальное видео
            st.markdown("**Исходное видео:**")
            with open(video_file_path, "rb") as vf:
                st.video(vf.read())

        else:
            # Режим покадрового просмотра
            frame_idx = st.session_state.current_frame
            bgr = read_frame(video_file_path, frame_idx)

            if bgr is not None:
                img = bgr
                yolo_count = None
                track_count = None

                # Применяем маски если чекбоксы активны
                if st.session_state.floor:
                    floor_mask = masks.get("floor")
                    if floor_mask is not None:
                        floor_config = masks_config["floor"]
                        img = apply_mask_to_frame(
                            img,
                            floor_mask,
                            color=floor_config["color"],
                            alpha=floor_config["alpha"]
                        )

                if st.session_state.window:
                    window_mask = masks.get("window")
                    if window_mask is not None:
                        window_config = masks_config["window"]
                        img = apply_mask_to_frame(
                            img,
                            window_mask,
                            color=window_config["color"],
                            alpha=window_config["alpha"]
                        )

                if st.session_state.yolo_enabled:
                    # читаем детекции с учетом фильтра уверенности и рисуем боксы
                    dets = get_frame_detections(
                        det_data,
                        frame_idx,
                        min_confidence=st.session_state.min_confidence
                    )
                    yolo_count = len(dets)
                    img = draw_bboxes_on_image(img, dets)

                if active_tracker_key is not None:
                    tracks = get_frame_tracks(tracks_data, frame_idx) if 'tracks_data' in locals() else []
                    track_count = len(tracks)

                    # Build short track history window for smooth trail drawing in frame-by-frame mode
                    history_len = 25
                    start_f = max(0, frame_idx - history_len + 1)
                    track_history = {}
                    if 'tracks_data' in locals():
                        for f in range(start_f, frame_idx + 1):
                            f_tracks = get_frame_tracks(tracks_data, f)
                            for tr in f_tracks:
                                tid = tr.get("id")
                                bbox = tr.get("bbox", {})
                                try:
                                    cx = int((bbox.get("x1", 0) + bbox.get("x2", 0)) / 2)
                                    cy = int(bbox.get("y2", 0))  # bottom center
                                except Exception:
                                    continue
                                if tid not in track_history:
                                    from collections import deque
                                    track_history[tid] = deque(maxlen=history_len)
                                track_history[tid].append((cx, cy))

                    img = draw_tracks_on_image(img, tracks, track_history)

                # Обувь: отрисовка сводки по кадру при активном флаге
                if st.session_state.shoe1:
                    try:
                        counts, avg_conf = summarize_frame_shoes(shoes_data, frame_idx)
                        img = draw_shoes_summary_on_image(img, counts, avg_conf)
                    except Exception:
                        pass

                rgb = img[:, :, ::-1]

                # Build caption
                parts = [f"Кадр {frame_idx}"]
                if st.session_state.yolo_enabled:
                    parts.append(f"YOLO: {yolo_count} (conf ≥ {st.session_state.min_confidence:.2f})")
                if active_tracker_key is not None:
                    parts.append(f"Tracks: {track_count}")
                caption = " — ".join(parts)

                st.image(rgb, caption=caption, use_container_width=True)
            else:
                st.warning("⚠️ Не удалось прочитать кадр")

        # Показать длительность, если смогли вычислить
        if st.session_state.video_duration:
            mins = st.session_state.video_duration // 60
            secs = st.session_state.video_duration % 60

            st.markdown(
                f"<p style='text-align: center; color: gray; font-size: 0.9em;'>"
                f"Длительность: {mins}:{secs:02d}"
                f"</p>",
                unsafe_allow_html=True
            )
    else:
        # Заглушка, если видео не найдено
        st.markdown(
            """
            <div style='background-color: #e9ecef; height: 400px; border-radius: 8px; 
            display: flex; align-items: center; justify-content: center; color: #6c757d;'>
                <div style='text-align: center;'>
                    <h2>📹 Video Not Found</h2>
                    <p>Поместите видеофайл в: data/raw/basketball_000.mp4</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Правая панель - Статистика
with col3:
    st.markdown("### Статистика")

    # Статистика видео
    if os.path.exists(video_file_path):
        try:
            video_info = get_video_info_safe(video_file_path)

            # Готовим безопасные значения
            width = int(video_info.get('width') or 0)
            height = int(video_info.get('height') or 0)
            fps_val = float(video_info.get('fps') or 0)
            duration_sec = int(video_info.get('duration') or 0)
            frames_stat = int(video_info.get('frame_count') or 0)

            # Среднее число детекций на кадр из JSON с учетом фильтра
            try:
                avg_det = compute_avg_detections(
                    det_data,
                    min_confidence=st.session_state.min_confidence if st.session_state.yolo_enabled else None
                ) if 'det_data' in locals() else 0.0
            except Exception:
                avg_det = 0.0

            # Среднее число треков на кадр из MOT-треков
            try:
                tracks_list = tracks_data.get("tracks", []) if 'tracks_data' in locals() else []
                if frames_stat and frames_stat > 0:
                    total_frames_for_avg = frames_stat
                else:
                    try:
                        max_frame_idx = max((int(t.get("frame", -1)) for t in tracks_list), default=-1)
                        total_frames_for_avg = max_frame_idx + 1 if max_frame_idx >= 0 else 0
                    except Exception:
                        total_frames_for_avg = 0
                avg_trk = (len(tracks_list) / total_frames_for_avg) if total_frames_for_avg > 0 else 0.0
            except Exception:
                avg_trk = 0.0

            # Форматирование
            fps_str = f"{fps_val:.2f}" if fps_val > 0 else "—"
            dur_str = f"{duration_sec} сек" if duration_sec > 0 else "—"
            res_str = f"{width} × {height}" if width > 0 and height > 0 else "—"
            frames_str = f"{frames_stat}" if frames_stat > 0 else "—"
            avg_det_str = f"{avg_det:.2f}" if avg_det > 0 else "—"
            avg_trk_str = f"{avg_trk:.2f}" if avg_trk > 0 else "—"

            # Динамически собираем HTML, скрывая отсутствующие поля
            rows = []
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>Разрешение:</strong>
                    <span style='float: right; color: #212529;'>{res_str}</span>
                </div>""")
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>FPS:</strong>
                    <span style='float: right; color: #212529;'>{fps_str}</span>
                </div>""")
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>Длительность:</strong>
                    <span style='float: right; color: #212529;'>{dur_str}</span>
                </div>""")
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>Кадров:</strong>
                    <span style='float: right; color: #212529;'>{frames_str}</span>
                </div>""")
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>Сред. детекций/кадр:</strong>
                    <span style='float: right; color: #212529;'>{avg_det_str}</span>
                </div>""")
            rows.append(f"""
                <div style='color: #6c757d; font-size: 0.875rem; margin-bottom: 0.5rem;'>
                    <strong>Сред. треков/кадр:</strong>
                    <span style='float: right; color: #212529;'>{avg_trk_str}</span>
                </div>""")

            html = "\n".join(rows)
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='stat-label'> Информация о видео</div>
                    <div style='margin-top: 0.75rem;'>
                        {html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown("""
                <div class='metric-card'>
                    <div class='stat-label'> Информация о видео</div>
                    <div style='color: #dc3545; font-size: 0.875rem; margin-top: 0.5rem;'>
                        Не удалось загрузить статистику
                    </div>
                </div>
            """, unsafe_allow_html=True)
    try:
        cur_f = int(st.session_state.get('current_frame', 0))
        st.markdown(f"""
                <div class='metric-card'>
                    <div style='text-align: center; color: #212529; font-weight: 600; margin-bottom: 0.5rem;'>🔘 Кадр: {cur_f} </div>
            """, unsafe_allow_html=True)
        # Детекции YOLO на текущем кадре


        cur_dets = get_frame_detections(
            det_data,
            cur_f,
            min_confidence=st.session_state.min_confidence if st.session_state.yolo_enabled else None
        ) if 'det_data' in locals() else []

        conf_info = f" (conf ≥ {st.session_state.min_confidence:.2f})" if st.session_state.yolo_enabled and st.session_state.min_confidence > 0 else ""

        st.markdown(f"""
            <div class='metric-card'>
                <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;'>
                    <div>YOLO детекций: {conf_info} <span style='float: right; color: #212529;'>{len(cur_dets)}</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    # Трекеры — количество треков на текущем кадре + расширенная статистика (показываем только при выбранном трекере)
    try:
        if 'active_tracker_key' in locals() and active_tracker_key is not None:
            cur_f_tr = int(st.session_state.get('current_frame', 0))
            cur_tracks = get_frame_tracks(tracks_data, cur_f_tr) if 'tracks_data' in locals() else []

            # Заголовок с числом треков на текущем кадре
            tracker_title = active_tracker_label + " треков" if active_tracker_label else "Треков"
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;'>
                        <div>{tracker_title}: <span style='float: right; color: #212529;'>{len(cur_tracks)}</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Дополнительная статистика по выбранному трекеру
            if 'tracks_data' in locals() and tracks_data.get('tracks'):
                tracks_list = tracks_data.get('tracks', [])
                # Подсчет уникальных ID и длины треков
                from collections import defaultdict
                lengths = defaultdict(int)
                for t in tracks_list:
                    try:
                        lengths[int(t.get('id'))] += 1
                    except Exception:
                        continue
                unique_ids = len(lengths)
                avg_len = (sum(lengths.values()) / unique_ids) if unique_ids > 0 else 0.0
                max_len = max(lengths.values()) if lengths else 0
                min_len = min(lengths.values()) if lengths else 0

                # Имя файла и префикс
                import os as _os
                source_path = tracks_data.get('meta', {}).get('source')
                base_name = _os.path.basename(source_path) if source_path else '—'
                prefix = base_name.split('_')[0] if base_name and '_' in base_name else '—'

                # Рендерим под строкой с количеством треков
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.25rem;'>
                            <div>Активный трекер: <span style='float: right; color: #212529;'>{active_tracker_label or '—'}</span></div>
                            <div>Файл разметки: <span style='float: right; color: #212529;'>{base_name}</span></div>
                            <div>Уникальных ID: <span style='float: right; color: #212529;'>{unique_ids}</span></div>
                            <div>Средняя длина трека (кадры): <span style='float: right; color: #212529;'>{avg_len:.2f}</span></div>
                            <div>Мин/Макс длина трека: <span style='float: right; color: #212529;'>{min_len} / {max_len}</span></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    except Exception:
        pass

    # Обувь на текущем кадре — вместо Note
    try:
        cur_f_sh = int(st.session_state.get('current_frame', 0))
        counts, avg_conf = summarize_frame_shoes(shoes_data if 'shoes_data' in locals() else {"labels": []}, cur_f_sh)
        if counts:
            # Формируем HTML-список: Класс — Кол-во (ср. уверенность)
            items = []
            for cls in sorted(counts.keys()):
                cnt = counts.get(cls, 0)
                conf = avg_conf.get(cls, None)
                if conf is not None:
                    items.append(f"<div>{cls}: <span style='float: right; color: #212529;'>{cnt} (avg {conf:.2f})</span></div>")
                else:
                    items.append(f"<div>{cls}: <span style='float: right; color: #212529;'>{cnt}</span></div>")
            items_html = "\n".join(items)
        else:
            items_html = "<div style='color: #6c757d;'>Нет данных об обуви на этом кадре</div>"

        st.markdown(f"""
            <div class='metric-card'>
                <div class='stat-label'>Обувь на кадре</div>
                <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;'>
                    {items_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
            <div class='metric-card'>
                <div class='stat-label'>Обувь на кадре</div>
                <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;'>
                    Нет данных
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Detect Id
    st.markdown("""
        <div class='metric-card'>
            <div class='stat-label' style='text-align: center; font-size: 1rem;'>Detect Id</div>
        </div>
    """, unsafe_allow_html=True)

    # Простая визуализация графика
    chart_data = pd.DataFrame(
        np.random.randn(20, 1) * 2 + 5,
        columns=['Connection (lick)']
    )
    st.bar_chart(chart_data, height=150)

    st.markdown("</div>", unsafe_allow_html=True)

    # Font by Liver
    st.markdown(
        "<div style='text-align: right; color: #6c757d; font-size: 0.75rem; margin-top: 0.5rem;'>Font by Liver</div>",
        unsafe_allow_html=True)

    # Процентные показатели
    perc_cols = st.columns(2)
    with perc_cols[0]:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-value' style='color: #6c757d;'>201%</div>
            </div>
        """, unsafe_allow_html=True)
    with perc_cols[1]:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-value' style='color: #28a745;'>94%</div>
            </div>
        """, unsafe_allow_html=True)