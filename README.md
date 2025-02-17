# PDF OCR Text Extraction Tool

This tool extracts text from scanned PDF documents where each page is an image. It uses Optical Character Recognition (OCR) to convert image-based PDFs into searchable text.

## Features

- Convert PDF pages to high-quality images
- Extract text using Tesseract OCR
- Support for multiple languages (with appropriate Tesseract language packs)
- Save individual page text files
- Combine all extracted text into a single output file
- Basic image preprocessing for better OCR accuracy

## Prerequisites

Before running this tool, you need to install:

### 1. Python Dependencies
```bash
pip install pdf2image pytesseract Pillow
```

### 2. Tesseract OCR

#### Windows:
1. Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to system PATH
3. Default installation path: `C:\Program Files\Tesseract-OCR`

#### Linux:
```bash
sudo apt-get install tesseract-ocr
```

#### MacOS:
```bash
brew install tesseract
```

### 3. Poppler

#### Windows:
1. Download from [Poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract to a folder
3. Add bin folder to system PATH

#### Linux:
```bash
sudo apt-get install poppler-utils
```

#### MacOS:
```bash
brew install poppler
```

## Usage

1. Basic usage:
```python
from pdf_extractor import extract_text_from_pdf

pdf_path = "path/to/your/scanned.pdf"
output_path = "path/to/output/extracted_text.txt"
extract_text_from_pdf(pdf_path, output_path)
```

2. With custom paths (Windows):
```python
pages = pdf2image.convert_from_path(
    pdf_path,
    poppler_path=r"C:\path\to\poppler\bin"
)
```

## Output Files

The tool generates:
- One text file per page: `output_path_page_1.txt`, `output_path_page_2.txt`, etc.
- Combined text file: `output_path.txt`

## Troubleshooting

1. "Unable to get page count. Is poppler installed and in PATH?"
   - Check if poppler is installed
   - Verify system PATH includes poppler
   - Try specifying poppler_path explicitly

2. "tesseract is not recognized"
   - Check if Tesseract is installed
   - Verify system PATH includes Tesseract
   - Try reinstalling Tesseract

3. Poor text recognition:
   - Check PDF quality
   - Try adjusting image preprocessing
   - Verify appropriate language packs are installed

## Performance Tips

1. For large PDFs:
   - Process pages in batches
   - Monitor memory usage
   - Consider parallel processing

2. For better accuracy:
   - Increase DPI during conversion
   - Adjust contrast and brightness
   - Clean up image noise

## Language Support

1. Install additional language packs:
   - Windows: Use Tesseract installer
   - Linux: `sudo apt-get install tesseract-ocr-[lang]`
   - MacOS: `brew install tesseract-lang`

2. Modify language in code:
```python
text = pytesseract.image_to_string(page, lang='fra')  # For French
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
