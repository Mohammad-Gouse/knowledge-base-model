import os
import docx

def extract_text_from_docx(doc_path):
    """Extracts text from a Word document."""
    doc = docx.Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_documents(folder_path):
    """Loads all Word documents from a folder and extracts text."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_docx(file_path)
            if text:
                documents.append({"id": filename, "text": text})
    return documents
