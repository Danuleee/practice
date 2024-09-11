from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import matplotlib.pyplot as plt

model = ocr_predictor(pretrained=True)

def visualize_ocr(result):
    synthetic_pages = result.synthesize()
    plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()

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
 
# 'from_images', 'from_pdf', 'from_url'
# doc = DocumentFile.from_url("https://nationalseniors.com.au/generated/1280w-3-2/auspassport-png.png?1677548746")
doc = DocumentFile.from_images("auspassport-png.png")
# Analyze
result = model(doc)

min_confidence = 0.9

# Filter the results
filtered_result = filter_low_confidence(result, min_confidence)

# Show the filtered results
filtered_result.show()


# visualize_ocr(filtered_result)