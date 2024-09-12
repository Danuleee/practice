import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from ocr import get_image_text
from classify import classify

def process_and_classify(features, classifier):    
    # Классифицируем
    class_probabilities = classifier(features)
    
    predicted_class = tf.argmax(class_probabilities, axis=-1)
    
    return predicted_class.numpy()[0]

# Использование
image_text = get_image_text("auspassport-png.png")
print(image_text)
document_class = classify(image_text)

print(f"Document class: {document_class}")
print("OCR result:")
print(document_class)