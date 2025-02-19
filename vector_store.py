from sentence_transformers import SentenceTransformer
from config import collection
from doc_processor import load_documents
from pdf_processor import load_pdf_documents

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks for better retrieval."""
    chunks = []
    words = text.split()  # Split by words to avoid breaking words
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def store_documents_from_word(doc_path):
    """Extracts text from Word docs, chunks it, converts to vectors, and stores in ChromaDB."""
    documents = load_documents(doc_path)  # Load Word documents

    for doc in documents:
        print("Chunking in progress...")
        text_chunks = chunk_text(doc["text"])  # Break document into smaller pieces

        for idx, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk).tolist()
            chunk_id = f"{doc['id']}_{idx}"  # Unique ID for each chunk
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )

    print("Word document stored in ChromaDB ✅")

def store_documents_from_pdf(pdf_path):
    """Extracts text from pdf docs, chunks it, converts to vectors, and stores in ChromaDB."""
    documents = load_pdf_documents(pdf_path)  # Load Word documents

    for doc in documents:
        text_chunks = chunk_text(doc["text"])  # Break document into smaller pieces

        for idx, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk).tolist()
            chunk_id = f"{doc['id']}_{idx}"  # Unique ID for each chunk
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )

    print("PDF document stored in ChromaDB ✅")

def search_knowledge_base(query, top_k=3):
    """Searches ChromaDB and returns the most relevant text chunks."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas"]
    )

    # print(len(results['metadatas'][0]))

    retrieved_texts = [item["text"] for item in results["metadatas"][0]] if results["metadatas"] else []

    return "\n".join(retrieved_texts) if retrieved_texts else None

