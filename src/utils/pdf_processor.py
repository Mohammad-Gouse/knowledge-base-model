import os

from src.utils.extract_text_pdf import extract_text_from_pdf


def load_pdf_documents(folder_path):
    """Loads all Word documents from a folder and extracts text."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path, f"../txts/{os.path.splitext(filename)[0]}")
            if text:
                documents.append({"id": filename, "text": text})
    return documents
