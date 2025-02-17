import pdf2image
import pytesseract
from PIL import Image
import os


def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extract text from a PDF containing scanned images using OCR.

    Args:
        pdf_path (str): Path to the input PDF file
        output_txt_path (str): Path where the extracted text will be saved
    """
    try:
        # Convert PDF to images
        # print("Converting PDF to images...")
        pages = pdf2image.convert_from_path(pdf_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        # Process each page
        all_text = []
        for i, page in enumerate(pages):
            print(f"Processing page {i + 1}/{len(pages)}...")

            # Improve image quality for better OCR results
            page = page.convert('L')  # Convert to grayscale

            # Optional: Improve image quality
            # page = page.point(lambda x: 0 if x < 128 else 255, '1')  # Increase contrast

            # Perform OCR
            text = pytesseract.image_to_string(page, lang='eng')
            all_text.append(text)

            # Optional: Save individual page text
            with open(f"{output_txt_path}_page_{i + 1}.txt", 'w', encoding='utf-8') as f:
                f.write(text)

        # Save all text to a single file
        with open(f"{output_txt_path}.txt", 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_text))

        print(f"Text extraction complete. Output saved to {output_txt_path}")
        return '\n\n'.join(all_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy.

    Args:
        image (PIL.Image): Input image
    Returns:
        PIL.Image: Processed image
    """
    # Convert to grayscale
    image = image.convert('L')

    # Increase contrast
    # image = image.point(lambda x: 0 if x < 128 else 255, '1')

    return image

