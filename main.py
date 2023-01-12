import pytesseract
import cv2
from PIL import Image
import numpy as np
import os
import re
from autocorrect import Speller

def process_image(image_path):
    # Open image file
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply image enhancement
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply image thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Save processed image
    cv2.imwrite("gray.png", gray)

    # Run image through OCR engine
    text = pytesseract.image_to_string(Image.open("gray.png"), lang='eng')

    # Perform post-processing
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII characters
    text = re.sub(r'[\n]+', '\n', text) # remove multiple newlines
    text = re.sub(r'\s+', ' ', text) # remove multiple spaces

    spell = Speller(lang='en')
    text = spell(text)
    # Save result to file
    with open("output.txt", "w") as file:
        file.write(text)
        
def main():
    # input image path
    img_path = "image.png"
    process_image(img_path)

if __name__ == "__main__":
    main()
