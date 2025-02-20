```md
# Project Setup Guide

This project requires specific Python packages along with **Poppler** and **Tesseract OCR** to function correctly. Follow the steps below to set up the environment.

## 1. Install Python Dependencies

Ensure you have **Python 3** and **pip** installed. Then, run:

```sh
pip install -r requirements.txt
```

This will install all required Python packages listed in `requirements.txt`.

## 2. Install Poppler

Poppler is required for handling PDF processing. Install it based on your operating system:

### Windows:
1. Download the latest **Poppler** binaries from [this link](https://github.com/oschwartz10612/poppler-windows/releases).
2. Extract the files and add the `bin` folder to the system **PATH** environment variable.

### macOS:
```sh
brew install poppler
```

### Linux (Debian/Ubuntu):
```sh
sudo apt update
sudo apt install poppler-utils
```

## 3. Install Tesseract OCR

Tesseract OCR is needed for Optical Character Recognition (OCR).

### Windows:
1. Download and install **Tesseract** from [this link](https://github.com/UB-Mannheim/tesseract/wiki).
2. Add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to the **PATH** environment variable.

### macOS:
```sh
brew install tesseract
```

### Linux (Debian/Ubuntu):
```sh
sudo apt update
sudo apt install tesseract-ocr
```

## 4. Verify Installations

Check if **Poppler** and **Tesseract OCR** are correctly installed by running:

```sh
pdftotext -v  # Check Poppler installation
tesseract -v  # Check Tesseract installation
```

## 5. Run the Project

After completing the setup, start the FastAPI application using **Uvicorn**:

```sh
uvicorn app:app --reload
```

- `app:app` → First `app` refers to the filename (e.g., `app.py`), and the second `app` is the FastAPI instance.
- `--reload` → Enables auto-reloading on code changes (useful for development).

Now, the API should be running on **http://127.0.0.1:8000/**.

---
