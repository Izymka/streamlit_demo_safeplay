import cv2
import numpy as np
import os


def load_mask(mask_path):
    """
    Загружает маску из файла
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    return mask


def apply_mask_to_frame(frame, mask, color=(0, 255, 0), alpha=0.3):
    """
    Накладывает маску на кадр с указанным цветом и прозрачностью
    """
    if mask is None:
        return frame

    # Приводим маску к формату RGBA (4 канала)
    if mask.ndim == 2:
        # Одноканальная маска (градации серого) — используем как альфа-канал
        h_m, w_m = mask.shape
        mask_rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
        mask_rgba[:, :, 3] = mask  # Alpha из исходной маски
    elif mask.ndim == 3:
        if mask.shape[2] == 1:
            # Одноканальная в виде (H, W, 1)
            alpha_channel = mask[:, :, 0]
            h_m, w_m = alpha_channel.shape
            mask_rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
            mask_rgba[:, :, 3] = alpha_channel
        elif mask.shape[2] == 3:
            # BGR -> BGRA
            mask_rgba = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
        elif mask.shape[2] == 4:
            # Уже BGRA
            mask_rgba = mask
        else:
            # Неожиданное число каналов — просто не накладываем маску
            return frame
    else:
        # Неподдерживаемая форма маски
        return frame

    # Создаем цветную маску на основе RGBA
    colored_mask = np.zeros_like(mask_rgba)
    colored_mask[:, :, 0] = color[0]  # B
    colored_mask[:, :, 1] = color[1]  # G
    colored_mask[:, :, 2] = color[2]  # R
    colored_mask[:, :, 3] = mask_rgba[:, :, 3]  # Alpha

    # Изменяем размер маски под размер кадра
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(colored_mask, (w, h))

    # Разделяем каналы
    mask_bgr = mask_resized[:, :, :3]
    mask_alpha = mask_resized[:, :, 3] / 255.0

    # Накладываем маску
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c] * (1 - mask_alpha * alpha) +
                          mask_bgr[:, :, c] * mask_alpha * alpha)

    return frame


def get_masks_config():
    """
    Возвращает конфигурацию масок
    """
    # Путь к корню проекта: utils/mask_utils.py -> utils -> <root>
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mask_dir = os.path.join(project_root, "assets", "mask")

    return {
        "floor": {
            "path": os.path.join(mask_dir, "floor_mask.png"),
            "color": (0, 255, 0),  # Зеленый
            "alpha": 0.3
        },
        "window": {
            "path": os.path.join(mask_dir, "window_mask.png"),  # именно window_mask.png
            "color": (255, 0, 0),  # Красный
            "alpha": 0.6
        }
    }