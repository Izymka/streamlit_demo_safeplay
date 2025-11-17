import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# utils
from utils.yolo_utils import (
    load_detections,
    get_frame_detections,
    compute_avg_detections,
    read_frame,
    draw_bboxes_on_image,
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

# Инициализация состояния приложения (безопасно по умолчанию)
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0

# Настройка страницы
st.set_page_config(
    page_title="Safe Play",
    page_icon="👥",
    layout="wide"
)

# Минималистичный стиль в светлых тонах
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

    st.markdown("**Детекции YOLO**")
    yolo_enabled = st.checkbox("Показывать детекции YOLO", value=True)

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    track_id = st.checkbox("Track ID ID", value=False)

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    st.markdown("**ROI Zone**")

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    shoe_classification_1 = st.checkbox("Shoe Classification", value=True, key="shoe1")

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    shoe_classification_2 = st.checkbox("Shoe Classification", value=True, key="shoe2")

    st.markdown("<hr style='margin:4px 0; opacity:0.3;'>", unsafe_allow_html=True)

    shoe_instecation = st.checkbox("Shoe Instecation", value=True)

# Центральная панель - Видео
with col2:
    st.markdown("### People Detection")

    # Пути к данным
    video_file_path = "data/raw/basketball_000.mp4"
    det_json_path = "assets/yolo_det/basketball_000.json"

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


    det_data = _load_json(det_json_path) if os.path.exists(det_json_path) else {"results": []}

    # Если число кадров неизвестно, попробуем взять из JSON
    if frames == 0:
        try:
            frames = int(det_data.get("video_info", {}).get("total_frames") or len(det_data.get("results", [])) or 0)
        except Exception:
            frames = len(det_data.get("results", []))

    # Селектор кадра (если есть хотя бы 1 кадр)
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
        # Всегда читаем конкретный кадр по индексу слайдера
        frame_idx = st.session_state.current_frame
        bgr = read_frame(video_file_path, frame_idx)

        if bgr is not None:
            if 'yolo_enabled' in locals() and yolo_enabled:
                # читаем детекции и рисуем боксы
                dets = get_frame_detections(det_data, frame_idx)
                bgr_drawn = draw_bboxes_on_image(bgr, dets)
                rgb = bgr_drawn[:, :, ::-1]
                caption = f"Кадр {frame_idx} — детекций: {len(dets)}"
            else:
                # показываем «чистый» кадр без детекций
                rgb = bgr[:, :, ::-1]
                caption = f"Кадр {frame_idx}"

            st.image(rgb, caption=caption, use_container_width=True)
        else:
            # Фолбэк, если кадр не удалось прочитать
            st.info("Не удалось прочитать кадр для отображения. Показываем видео.")
            with open(video_file_path, "rb") as vf:
                st.video(vf.read())

        # Показать длительность, если смогли вычислить
        if st.session_state.video_duration:
            mins = st.session_state.video_duration // 60
            secs = st.session_state.video_duration % 60
            st.caption(f"Длительность: {mins}:{secs:02d}")
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
    st.markdown("### 📊 Статистика")

    # Статистика видео
    if os.path.exists(video_file_path):
        try:
            video_info = get_video_info_safe(video_file_path)

            # Готовим безопасные значения
            width = int(video_info.get('width') or 0)
            height = int(video_info.get('height') or 0)
            fps = float(video_info.get('fps') or 0)
            duration_sec = int(video_info.get('duration') or 0)
            frames_stat = int(video_info.get('frame_count') or 0)

            # Среднее число детекций на кадр из JSON
            try:
                avg_det = compute_avg_detections(det_data) if 'det_data' in locals() else 0.0
            except Exception:
                avg_det = 0.0

            # Форматирование
            fps_str = f"{fps:.2f}" if fps > 0 else "—"
            dur_str = f"{duration_sec} сек" if duration_sec > 0 else "—"
            res_str = f"{width} × {height}" if width > 0 and height > 0 else "—"
            frames_str = f"{frames_stat}" if frames_stat > 0 else "—"
            avg_det_str = f"{avg_det:.2f}" if avg_det > 0 else "—"

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

            html = "\n".join(rows)
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='stat-label'>📹 Информация о видео</div>
                    <div style='margin-top: 0.75rem;'>
                        {html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown("""
                <div class='metric-card'>
                    <div class='stat-label'>📹 Информация о видео</div>
                    <div style='color: #dc3545; font-size: 0.875rem; margin-top: 0.5rem;'>
                        Не удалось загрузить статистику
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Детекции YOLO (вместо "Şiramin")
    try:
        cur_f = int(st.session_state.get('current_frame', 0))
        cur_dets = get_frame_detections(det_data, cur_f) if 'det_data' in locals() else []
        st.markdown(f"""
            <div class='metric-card'>
                <div class='stat-label'>Детекции YOLO</div>
                <div style='color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;'>
                    <div style='margin-bottom: 0.25rem;'>Кадр: <span style='float: right; color: #212529;'>{cur_f}</span></div>
                    <div>Количество детекций: <span style='float: right; color: #212529;'>{len(cur_dets)}</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    # Duse type
    st.markdown("""
        <div class='metric-card'>
            <div class='stat-label'>Duse type</div>
            <div class='stat-value'>81,7.36.15</div>
            <div style='margin-top: 0.5rem;'>
                <div style='color: #6c757d; font-size: 0.75rem;'>Video: <span style='float: right;'>27,74 %</span></div>
                <div style='color: #6c757d; font-size: 0.75rem;'>1Type: <span style='float: right;'>8.37,75</span></div>
                <div style='color: #6c757d; font-size: 0.75rem;'>Notes: <span style='float: right;'>1,111</span></div>
                <div style='color: #6c757d; font-size: 0.75rem;'>Kib: <span style='float: right;'>-80.19</span></div>
                <div style='color: #6c757d; font-size: 0.75rem;'>ID: <span style='float: right;'>∗</span></div>
                <div style='color: #6c757d; font-size: 0.75rem;'>Skl: <span style='float: right;'>1.25,50</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Note
    st.markdown("""
        <div class='metric-card'>
            <div class='stat-label'>Note:</div>
            <div style='color: #212529; font-size: 1rem;'>80.4; 135.ikm 🔵</div>
        </div>
    """, unsafe_allow_html=True)

    # Detect Id
    st.markdown("""
        <div class='metric-card'>
            <div class='stat-label' style='text-align: center; font-size: 1rem;'>Detect Id</div>
        </div>
    """, unsafe_allow_html=True)

    # График
    st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center; color: #212529; font-weight: 600; margin-bottom: 0.5rem;'>1DI</div>
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

# Кнопка паузы в правом верхнем углу
st.markdown("""
    <div style='position: fixed; top: 1rem; right: 1rem; z-index: 999;'>
        <button style='background-color: white; border: 1px solid #dee2e6; border-radius: 50%; 
        width: 40px; height: 40px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            ⏸️
        </button>
    </div>
""", unsafe_allow_html=True)