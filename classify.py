from transformers import LayoutLMForSequenceClassification, LayoutLMTokenizer
from PIL import Image
import torch

def classify(text):
    # Load LayoutLM tokenizer and model
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

    # Tokenize the text and feed it into the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass to get predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to class predictions
    predicted_class = torch.argmax(logits, dim=1).item()

    print(f"Predicted class: {predicted_class}")


if __name__  == "__main__":
    a = classify("passport")
    print(a)