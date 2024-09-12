import cv2
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection_predictor

# Инициализация моделей OCR и детекции
ocr_model = ocr_predictor(pretrained=True)
detect_model = detection_predictor('db_resnet50', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)

def preprocess_image(image_path):
    # Считываем изображение с помощью OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Изображение не найдено или не удается загрузить: {image_path}")
    
    # Преобразование в оттенки серого
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Коррекция контраста с гистограммным выравниванием
    contrast_img = cv2.equalizeHist(gray_img)
    
    # Уменьшение яркости белых областей и увеличение насыщенности черных
    # Мы будем использовать операции инверсии и усиления контраста
    
    # Инверсия изображения (чтобы сделать белые области черными и наоборот)
    inverted_img = cv2.bitwise_not(contrast_img)
    
    # Увеличение контраста для инвертированного изображения
    alpha = 1.5  # Коэффициент контраста
    beta = 0     # Смещение яркости
    
    # Применение преобразования контраста
    contrasted_img = cv2.convertScaleAbs(inverted_img, alpha=alpha, beta=beta)
    
    # Обратное преобразование (инверсия снова)
    final_img = cv2.bitwise_not(contrasted_img)
    
    # Сохранение обработанного изображения
    processed_image_path = 'processed_' + image_path
    cv2.imwrite(processed_image_path, final_img)
    
    return processed_image_path

def visualize_ocr(result):
    synthetic_pages = result.synthesize()
    plt.imshow(synthetic_pages[0])
    plt.axis('off')
    plt.show()

def filter_low_confidence(result, min_confidence):
    filtered_pages = []
    for page in result.pages:
        filtered_blocks = []
        for block in page.blocks:
            filtered_lines = []
            for line in block.lines:
                filtered_words = [word for word in line.words if word.confidence >= min_confidence]
                if filtered_words:
                    line.words = filtered_words
                    filtered_lines.append(line)
            if filtered_lines:
                block.lines = filtered_lines
                filtered_blocks.append(block)
        page.blocks = filtered_blocks
        filtered_pages.append(page)
    result.pages = filtered_pages
    return result

# Обработка изображения
image_path = "auspassport-png.jpg"
processed_image_path = preprocess_image(image_path)

# Загрузка и анализ документа
doc = DocumentFile.from_images(processed_image_path)
result = ocr_model(doc)

# Установка минимального уровня уверенности для фильтрации
min_confidence = 0.4

# Фильтрация результатов на основе уверенности
filtered_result = filter_low_confidence(result, min_confidence)

# Отображение отфильтрованных результатов
filtered_result.show()

# По желанию можно визуализировать результаты OCR
visualize_ocr(filtered_result)
