import boto3
import chromadb
import json
import os
import docx
import pdf2image
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from typing import Dict, Optional, List
import httpx
from dotenv import load_dotenv


# Initialize ChromaDB (Persistent storage)
CHROMA_DB_PATH = "embeddings"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="knowledge_base")

# Initialize Amazon Bedrock Client
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")



def ask_bedrock(query, context):
    """Uses Amazon Bedrock to generate an answer based on retrieved knowledge"""
    # prompt_data = f"Answer the following question based on the provided knowledge: {context} \n\n Question: {query}"
    prompt_data = (
        "You are an intelligent and friendly assistant. Answer the following question "
        "in a conversational and natural tone based on the provided knowledge. "
        "Keep your response engaging and easy to understand.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"User question: {query}\n"
    )

    payload = {
        "prompt": "<s>[INST]" + prompt_data + "[/INST]",
        "max_tokens": int(os.getenv("MAX_TOKENS", 300)),
        "temperature": float(os.getenv("TEMPERATURE", 0.5)),
        "top_p": float(os.getenv("TOP_P", 0.9)),
        "top_k": float(os.getenv("TOP_K", 50)),
    }

    body = json.dumps(payload)
    model_id = "mistral.mistral-7b-instruct-v0:2"
    respone = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    response_body = json.loads(respone.get("body").read())
    return response_body


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


def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extract text from a PDF containing scanned images using OCR.

    Args:
        pdf_path (str): Path to the input PDF file
        output_txt_path (str): Path where the extracted text will be saved
    """
    try:
        # Convert PDF to images
        pages = pdf2image.convert_from_path(pdf_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        # Process each page
        all_text = []
        for i, page in enumerate(pages):
            print(f"Processing page {i + 1}/{len(pages)}...")

            # Improve image quality for better OCR results
            page = page.convert("L")  # Convert to grayscale

            # Optional: Improve image quality
            # page = page.point(lambda x: 0 if x < 128 else 255, '1')  # Increase contrast

            # Perform OCR
            text = pytesseract.image_to_string(page, lang="eng")
            all_text.append(text)

            # Optional: Save individual page text
            with open(
                f"{output_txt_path}_page_{i + 1}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(text)

        # Save all text to a single file
        with open(f"{output_txt_path}.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_text))

        print(f"Text extraction complete. Output saved to {output_txt_path}")
        return "\n\n".join(all_text)

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
    image = image.convert("L")

    # Increase contrast
    # image = image.point(lambda x: 0 if x < 128 else 255, '1')

    return image


def load_pdf_documents(folder_path):
    """Loads all Word documents from a folder and extracts text."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(
                file_path, f"txts/{os.path.splitext(filename)[0]}"
            )
            if text:
                documents.append({"id": filename, "text": text})
    return documents




def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks for better retrieval."""
    chunks = []
    words = text.split()  # Split by words to avoid breaking words
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
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
                ids=[chunk_id], embeddings=[embedding], metadatas=[{"text": chunk}]
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
                ids=[chunk_id], embeddings=[embedding], metadatas=[{"text": chunk}]
            )

    print("PDF document stored in ChromaDB ✅")


def search_knowledge_base(query, top_k=2, score_threshold=1.9):
    """Searches ChromaDB and returns the most relevant text chunks.
    If the ranking score is too low, return 'Not found' or None.
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or not results["distances"]:
        return None

    # Get the highest-ranked result's score (assuming lower distance = better match)
    best_score = results["distances"][0][0] if results["distances"][0] else float("inf")

    # If the score is too low (high distance), return "Not found"
    if best_score > score_threshold:
        return None

    # Extract the relevant texts
    retrieved_texts = [item["text"] for item in results["metadatas"][0]]

    return "\n".join(retrieved_texts) if retrieved_texts else None


# def store_by_document_type():
#     doc_path = "docs/"
#     pdf_path = "pdfs/"
#     doc_type = input("Enter Document type PDF or Word? [P/W]:")
#     if doc_type.lower() == "w" or doc_type.lower() == "word":
#         store_documents_from_word(doc_path)
#     elif doc_type.lower() == "p" or doc_type.lower() == "pdf":
#         store_documents_from_pdf(pdf_path)
#     else:
#         try_again = input("Invalid input. Do you want to try again?[Y/N]:")
#         if try_again.lower() == "y" or try_again.lower() == "yes":
#             store_by_document_type()
#         else:
#             return
#
#     more_doc = input("Do you want to store more documents? [Y/N]:")
#     if more_doc.lower() == "y" or more_doc.lower() == "yes":
#         store_by_document_type()


# def store_documents():
#     # Step 1: Store documents in ChromaDB
#     store_doc = input("Do you want to store documents? [Y/N]:")
#     if store_doc.lower() == "y" or store_doc.lower() == "yes":
#         store_by_document_type()


def search_user_query(query):
    # Query Examples:
    # query = "What is the setup cost for this mandate?"
    # query = "What is name of client and his pan number and his address, also on which date this mandate was signed?"
    # query1 = "how many Required fields that will be fetched from Caliber during integration of non functional requirements?"
    # query2 = "How to set up family trust?"

    threshold = float(os.getenv("THRESHOLD"))
    num_chunks = int(os.getenv("CHUNKS"))
    retrieved_text = search_knowledge_base(
        query, top_k=num_chunks, score_threshold=threshold
    )

    if retrieved_text:
        response = ask_bedrock(query, retrieved_text)
        return response["outputs"][0]["text"]
    else:
        print("No relevant data found.")
        return None


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


async def process_document(file: UploadFile) -> Dict:
    """Process uploaded document and store in ChromaDB."""

    # Define the temporary directory and ensure it exists
    TEMP_DIR = "/tmp"
    os.makedirs(TEMP_DIR, exist_ok=True)  # ✅ Create /tmp/ if it doesn't exist

    # Define the temporary file path
    temp_path = os.path.join(TEMP_DIR, file.filename)

    try:
        # Save the uploaded file
        content = await file.read()
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)

        # Debugging - Check if the file exists
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=500, detail=f"File not found: {temp_path}")

        # Extract text based on file type
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(
                temp_path, f"/tmp/{os.path.splitext(file.filename)[0]}"
            )
        elif file.filename.lower().endswith(".docx"):
            text = extract_text_from_docx(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file.filename}")

        # Chunk and store in ChromaDB
        text_chunks = chunk_text(text)
        for idx, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk).tolist()
            chunk_id = f"{file.filename}_{idx}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk, "source": file.filename}],
            )

        return {"message": f"File {file.filename} saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup - Delete file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)


# async def process_document(file: UploadFile) -> Dict:
#     """Process uploaded document and store in ChromaDB."""
#     temp_path = f"/tmp/{file.filename}"
#     try:
#         # Save uploaded file temporarily
#         with open(temp_path, "wb") as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#
#         # Extract text based on file type
#         if file.filename.lower().endswith('.pdf'):
#             text = extract_text_from_pdf(temp_path, f"/tmp/{os.path.splitext(file.filename)[0]}")
#         elif file.filename.lower().endswith('.docx'):
#             text = extract_text_from_docx(temp_path)
#         else:
#             raise ValueError(f"Unsupported file type: {file.filename}")
#
#         # Chunk and store in ChromaDB
#         text_chunks = chunk_text(text)
#         for idx, chunk in enumerate(text_chunks):
#             embedding = model.encode(chunk).tolist()
#             chunk_id = f"{file.filename}_{idx}"
#             collection.add(
#                 ids=[chunk_id],
#                 embeddings=[embedding],
#                 metadatas=[{"text": chunk, "source": file.filename}]
#             )
#
#         # Upload to S3 if configured
#         # if S3_BUCKET:
#         #     s3_client.upload_file(temp_path, S3_BUCKET, file.filename)
#
#         return {"message": f"Successfully processed {file.filename}", "chunks": len(text_chunks)}
#
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Endpoint to upload and process documents."""
    try:
        result = await process_document(file)
        return result
    except Exception as e:
        # logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_knowledge(request: QueryRequest):
    """API to query the knowledge base and get responses."""
    query = request.query
    threshold = float(os.getenv("THRESHOLD"))
    num_chunks = int(os.getenv("CHUNKS"))
    retrieved_text = search_knowledge_base(
        query, top_k=num_chunks, score_threshold=threshold
    )

    if not retrieved_text:
        return {"response": "No relevant data found"}

    response = ask_bedrock(query, retrieved_text)
    return {"response": response["outputs"][0]["text"]}


# In-memory cache to track processed messages (Prevent duplicates)
processed_messages = set()


@app.get("/")
def home():
    return {"message": "WhatsApp Webhook Running!..."}


@app.get("/webhook")
async def webhook(request: Request):
    query_params = request.query_params
    mode = query_params.get("hub.mode")
    challenge = query_params.get("hub.challenge")
    token = query_params.get("hub.verify_token")

    if mode and token:
        if mode == "subscribe" and token == MY_TOKEN:
            return int(challenge)  # FastAPI automatically sets 200 status
        else:
            return {"error": "Forbidden"}, 403


@app.post("/webhook")
async def webhook(request: Request):
    body_param = await request.json()
    access_token = ACCESS_TOKEN

    if body_param.get("object"):
        try:
            entry = body_param.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])

            if messages:
                phon_no_id = value["metadata"]["phone_number_id"]
                from_ = messages[0]["from"]
                msg_body = messages[0]["text"]["body"]
                user_name = (
                    value.get("contacts", [{}])[0]
                    .get("profile", {})
                    .get("name", "User")
                )

                answer = ""

                if search_user_query(msg_body) is not None:
                    answer = search_user_query(msg_body)
                else:
                    answer = "Sorry i am not able to answer your query"

                # Making the API call to WhatsApp Business API
                url = f"https://graph.facebook.com/v21.0/{phon_no_id}/messages?access_token={access_token}"
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": "8692809476",
                    "type": "text",
                    "text": {"body": f"{answer}"},
                }
                headers = {"Content-Type": "application/json"}

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        print("Message sent successfully:", response.json())
                        return {"message": "Message sent successfully"}
                    else:
                        print("Error sending message:", response.json())
                        return {
                            "error": "Failed to send message.",
                            "details": response.json(),
                        }, 400
        except Exception as e:
            print("Unexpected error:", str(e))
            return {"error": "An unexpected error occurred.", "details": str(e)}, 500
    return {"error": "Invalid request"}, 404



def get_auto_response(user_message, body_param):
    """Calls AWS Lambda to get an auto-response"""
    try:
        lambda_client = boto3.client("lambda")
        lambda_function_name = "testFunction"

        response = lambda_client.invoke(
            FunctionName=lambda_function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps({"userMessage": user_message, "bodyParam": body_param}),
        )

        lambda_response = json.loads(response["Payload"].read())
        return lambda_response.get("body", "Sorry, I didn't understand your request.")
    except Exception as e:
        print("Error calling Lambda:", e)
        return "Sorry, something went wrong."


def send_whatsapp_message(phone_number_id, to, message):
    """Sends message via WhatsApp API"""
    # whatsapp_token = os.getenv("WHATSAPP_TOKEN")
    whatsapp_token = "EAAWA3MqEIZBMBO2jAXsVW1fWrBq6o0NZBjybzl6nGYnwl9VkEbRa3MWIQgQjFqZBwJRtkdUOt0Bq2V4ADQzT1RotjM24xTNiAzlartIlH2ftPkKqNf57b3oyfl5aBhRiGhzZBNbiBCIhOKZAZAbJXz6K0ao9D3rLWPvKgvIkZBHKvKvsSWESKK3z5hDFUdMJPnZAUa09EswpBqNKvZBcuzo9QklRUXQZB2DriZBf5myRUGx"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {whatsapp_token}",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message},
    }
    whatsapp_api_url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"

    try:
        response = requests.post(whatsapp_api_url, json=payload, headers=headers)
        print("WhatsApp API Response:", response.json())
    except Exception as e:
        print("Error sending message:", e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# def lambda_handler(event, context):
#     """
#     AWS Lambda entry point.
#     Expects a JSON event with a "query" key.
#     """
#     try:
#         # Extract query from the event
#         user_query = event.get("query", "No query provided")
#
#         # Process the query
#         response_message = search_user_query(user_query)
#
#         return {
#             "statusCode": 200,
#             "body": json.dumps({"response": response_message})
#         }
#     except Exception as e:
#         return {
#             "statusCode": 500,
#             "body": json.dumps({"error": str(e)})
#         }
