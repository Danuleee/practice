import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection_predictor
from classify import DocumentClassifier  # Импортируем класс для классификации
warnings.filterwarnings("ignore")
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
    
    # Применение CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_img)
    
    # Инвертирование изображения
    inverted_img = cv2.bitwise_not(contrast_img)
    
    # Применение преобразования контраста (можно настроить alpha и beta)
    alpha = 1.0  # Коэффициент контраста (попробуйте разные значения)
    beta = 0      # Смещение яркости (попробуйте разные значения)
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
image_path = "Birth-Certificate-Template-1-TemplateLab-scaled.png"
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

# Извлечение текста с помощью OCR
ocr_text = " ".join([word.value for page in filtered_result.pages for block in page.blocks for line in block.lines for word in line.words])

# Классификация документа с помощью DocumentClassifier
classifier = DocumentClassifier()
document_type = classifier.classify_document(ocr_text)

# Выводим результат классификации
print(f"Тип документа: {document_type}")

# По желанию можно визуализировать результаты OCR
visualize_ocr(filtered_result)