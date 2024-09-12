import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from ocr import get_image_text

def create_classifier(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2048,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

classifier = create_classifier(num_classes=5)

def process_and_classify(features, classifier):    
    # Классифицируем
    class_probabilities = classifier(features)
    
    predicted_class = tf.argmax(class_probabilities, axis=-1)
    
    return predicted_class.numpy()[0]

# Использование
image_text = get_image_text("auspassport-png.png")
document_class = process_and_classify(image_text, classifier)

print(f"Document class: {document_class}")
print("OCR result:")
print(document_class)