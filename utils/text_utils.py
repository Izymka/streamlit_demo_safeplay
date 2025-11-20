import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Кеш для шрифтов, чтобы не загружать их каждый кадр
_fonts = {}


def get_pil_font(font_height):
    # Путь к шрифту. Убедитесь, что этот файл существует и поддерживает кириллицу!
    # Если файла нет, попробуйте заменить на системный путь или скачайте .ttf файл в assets/
    font_path = "assets/DejaVuSans.ttf"

    key = (font_path, font_height)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(font_path, font_height)
        except IOError:
            # Если шрифт не найден, используем дефолтный (кириллица может не работать)
            print(f"Warning: Font {font_path} not found. Using default PIL font.")
            _fonts[key] = ImageFont.load_default()

    return _fonts[key]


def draw_text(img, text, org, font_height=20, color=(255, 255, 255), thickness=-1):
    """
    Рисует текст с поддержкой UTF-8 (кириллицы) через Pillow.
    :param img: numpy array (BGR изображение от OpenCV)
    :param text: текст для отрисовки
    :param org: (x, y) координаты нижнего левого угла текста (как в cv2.putText)
    :param font_height: высота шрифта
    :param color: цвет текста (B, G, R) - формат OpenCV
    :param thickness: толщина (в PIL игнорируется или эмулируется, здесь пропущен для простоты)
    """
    # 1. Конвертируем BGR (OpenCV) -> RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = get_pil_font(font_height)

    # 2. Вычисляем размер текста для корректировки координат
    # PIL рисует от верхнего левого угла, а OpenCV принимает нижний левый (baseline)
    bbox = font.getbbox(text)  # (left, top, right, bottom)
    text_h = bbox[3] - bbox[1]

    x, y = org
    # Поднимаем Y на высоту текста, чтобы эмулировать поведение OpenCV (org = bottom-left)
    # (Небольшая корректировка font_height * 0.2 для учета "хвостиков" букв типа 'щ', 'р', 'у')
    y_top = y - text_h - int(font_height * 0.2)

    # 3. Рисуем текст. PIL требует RGB, а входной color в BGR.
    b, g, r = color
    draw.text((x, y_top), text, font=font, fill=(r, g, b))

    # 4. Конвертируем обратно RGB -> BGR и обновляем исходный массив in-place
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def text_size(text, font_height=20):
    """
    Возвращает ширину и высоту текста.
    """
    font = get_pil_font(font_height)
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height
