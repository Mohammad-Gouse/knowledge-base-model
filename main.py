# Import organization: Standard library, third-party, and local imports grouped
import json
import os
from typing import Dict, List, Any, Optional

# Third-party imports
import boto3
import chromadb
import docx
import httpx
import pdf2image
import pytesseract
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks, Query
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import tempfile
from urllib.parse import unquote_plus




# Configuration and environment setup
load_dotenv()

# Constants
CHROMA_DB_PATH = "/data/chromadb"
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
THRESHOLD = float(os.getenv("THRESHOLD", "1.9"))
NUM_CHUNKS = int(os.getenv("CHUNKS", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
TOP_P = float(os.getenv("TOP_P", 0.9))
TOP_K = float(os.getenv("TOP_K", 50))
TEMP_DIR = "/tmp"
FILE_KEY = "allowed_numbers.json"

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET_NAME")  # Add this to your .env file

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)


# Ensure temporary directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize ChromaDB
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="knowledge_base")

# Initialize Amazon Bedrock Client
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

# Load the embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Load embedding model (cache it to avoid reloading)
MODEL_PATH = "/data/model_cache"
os.makedirs(MODEL_PATH, exist_ok=True)

if not os.path.exists(os.path.join(MODEL_PATH, "all-MiniLM-L6-v2")):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model.save(MODEL_PATH)
else:
    embedding_model = SentenceTransformer(MODEL_PATH)

# In-memory cache to track processed messages
processed_messages = set()

# API models
class QueryRequest(BaseModel):
    query: str


# Document processing functions
def extract_text_from_docx(doc_path: str) -> str:
    """
    Extract text from a Word document.
    
    Args:
        doc_path: Path to the Word document
        
    Returns:
        Extracted text as a string
    """
    doc = docx.Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: Input image
        
    Returns:
        Processed image
    """
    return image.convert("L")  # Convert to grayscale


def extract_text_from_pdf(pdf_path: str, output_txt_path: str) -> str:
    """
    Extract text from a PDF containing scanned images using OCR.
    
    Args:
        pdf_path: Path to the input PDF file
        output_txt_path: Path where the extracted text will be saved
        
    Returns:
        Extracted text as a string
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
            page = preprocess_image(page)
            
            # Perform OCR
            text = pytesseract.image_to_string(page, lang="eng")
            all_text.append(text)
            
            # Optional: Save individual page text
            with open(f"{output_txt_path}_page_{i + 1}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        
        # Save all text to a single file
        combined_text = "\n\n".join(all_text)
        with open(f"{output_txt_path}.txt", "w", encoding="utf-8") as f:
            f.write(combined_text)
        
        print(f"Text extraction complete. Output saved to {output_txt_path}")
        return combined_text
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: Input text to be chunked
        chunk_size: Maximum number of words per chunk
        overlap: Number of overlapping words between consecutive chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    words = text.split()  # Split by words to avoid breaking words
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# Vector database operations
def store_document_in_chromadb(filename: str, text: str) -> Dict:
    """
    Store document text in ChromaDB after chunking and embedding.
    
    Args:
        filename: Name of the source file
        text: Extracted text from the document
        
    Returns:
        Dictionary with status message and chunk count
    """
    text_chunks = chunk_text(text)
    
    for idx, chunk in enumerate(text_chunks):
        embedding = embedding_model.encode(chunk).tolist()
        chunk_id = f"{filename}_{idx}"
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[{"text": chunk, "source": filename}]
        )
    
    return {"message": f"Document '{filename}' stored successfully", "chunks": len(text_chunks)}


def search_knowledge_base(query: str, top_k: int = 2, score_threshold: float = 1.9) -> Optional[str]:
    """
    Search ChromaDB and return the most relevant text chunks.
    
    Args:
        query: The search query
        top_k: Number of results to retrieve
        score_threshold: Maximum distance threshold for relevance
        
    Returns:
        Joined text of relevant chunks or None if no relevant results
    """
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    
    if not results["metadatas"] or not results["distances"]:
        return None
    
    # Get the highest-ranked result's score (lower distance = better match)
    best_score = results["distances"][0][0] if results["distances"][0] else float("inf")
    
    # If the score is too low (high distance), return None
    if best_score > score_threshold:
        return None
    
    # Extract the relevant texts
    retrieved_texts = [item["text"] for item in results["metadatas"][0]]
    
    return "\n".join(retrieved_texts) if retrieved_texts else None


# LLM interaction
def ask_bedrock(query: str, context: str) -> Dict:
    """
    Use Amazon Bedrock to generate an answer based on retrieved knowledge.
    
    Args:
        query: User's question
        context: Retrieved context from knowledge base
        
    Returns:
        Bedrock LLM response
    """
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
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K
    }
    
    body = json.dumps(payload)
    model_id = "mistral.mistral-7b-instruct-v0:2"
    
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    
    return json.loads(response.get("body").read())


def search_user_query(query: str) -> Optional[str]:
    """
    Process a user query - search the knowledge base and get LLM response.
    
    Args:
        query: User's question
        
    Returns:
        Generated answer or None if no relevant information found
    """
    retrieved_text = search_knowledge_base(
        query, 
        top_k=NUM_CHUNKS, 
        score_threshold=THRESHOLD
    )
    
    if retrieved_text:
        response = ask_bedrock(query, retrieved_text)
        return response["outputs"][0]["text"]
    else:
        print("No relevant data found.")
        return None


# WhatsApp integration
def send_whatsapp_message(phone_number_id: str, to: str, message: str) -> None:
    """
    Send a message via WhatsApp Business API.
    
    Args:
        phone_number_id: WhatsApp Business phone number ID
        to: Recipient's phone number
        message: Message text to send
    """
    whatsapp_token = os.getenv("WHATSAPP_TOKEN", ACCESS_TOKEN)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {whatsapp_token}"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    
    whatsapp_api_url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
    
    try:
        response = requests.post(whatsapp_api_url, json=payload, headers=headers)
        print("WhatsApp API Response:", response.json())
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

def list_s3_documents(
    bucket_name: str, 
    prefix: Optional[str] = None, 
    file_extensions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    List documents in an S3 bucket with optional filtering.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional S3 path prefix to filter objects
        file_extensions: Optional list of file extensions to filter (e.g., ['.pdf', '.docx'])
        
    Returns:
        List of dictionaries containing document information
    """
    if file_extensions is None:
        file_extensions = ['.pdf', '.docx']
        
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        params = {'Bucket': bucket_name}
        if prefix:
            params['Prefix'] = prefix
            
        document_list = []
        
        for page in paginator.paginate(**params):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Check if file has an allowed extension
                if any(key.lower().endswith(ext) for ext in file_extensions):
                    document_list.append({
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'type': 'pdf' if key.lower().endswith('.pdf') else 'docx'
                    })
                    
        return document_list
        
    except Exception as e:
        print(f"Error listing S3 objects: {str(e)}")
        raise


async def process_s3_document(bucket_name: str, key: str) -> Dict:
    """
    Process a document from S3 bucket and store in ChromaDB.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key (path to file)
        
    Returns:
        Dictionary with processing results
    """
    # Create a temporary file to store the downloaded document
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Download file from S3
        s3_client.download_file(bucket_name, key, temp_path)
        
        # Extract text based on file type
        if key.lower().endswith('.pdf'):
            text = extract_text_from_pdf(
                temp_path, 
                f"{TEMP_DIR}/{os.path.splitext(os.path.basename(key))[0]}"
            )
        elif key.lower().endswith('.docx'):
            text = extract_text_from_docx(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {key}")
        
        # Store in ChromaDB
        text_chunks = chunk_text(text)
        for idx, chunk in enumerate(text_chunks):
            embedding = embedding_model.encode(chunk).tolist()
            chunk_id = f"s3_{bucket_name}_{key.replace('/', '_')}_{idx}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{
                    "text": chunk,
                    "source": key,
                    "bucket": bucket_name
                }]
            )
        
        return {
            "message": f"Successfully processed S3 document: {key}",
            "chunks": len(text_chunks),
            "source": "s3",
            "bucket": bucket_name,
            "key": key
        }
        
    except Exception as e:
        print(f"Error processing S3 document {key}: {str(e)}")
        raise
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def process_s3_documents_batch(
    bucket_name: str, 
    keys: List[str],
    background_tasks: BackgroundTasks
) -> Dict:
    """
    Queue multiple S3 documents for background processing.
    
    Args:
        bucket_name: Name of the S3 bucket
        keys: List of S3 object keys to process
        background_tasks: FastAPI background tasks object
        
    Returns:
        Status message
    """
    for key in keys:
        # Add each document processing task to background tasks
        background_tasks.add_task(process_s3_document, bucket_name, key)
        
    return {
        "message": f"Processing {len(keys)} documents in the background",
        "documents": keys
    }

async def process_all_s3_documents(
    bucket: str,
    prefix: Optional[str] = None,
    max_documents: Optional[int] = None,
    file_extensions: Optional[List[str]] = None
) -> Dict:
    """
    Process all documents in an S3 bucket matching the criteria.
    
    Args:
        bucket: Name of the S3 bucket
        prefix: Optional prefix to filter documents (folder path)
        max_documents: Optional maximum number of documents to process
        file_extensions: Optional list of file extensions to include
        
    Returns:
        Dictionary with processing results
    """
    documents = list_s3_documents(bucket, prefix, file_extensions)
    processed_count = 0
    results = []
    
    # Apply max_documents limit if specified
    if max_documents is not None:
        documents = documents[:max_documents]
    
    for doc in documents:
        try:
            result = await process_s3_document(bucket, doc['key'])
            results.append(result)
            processed_count += 1
        except Exception as e:
            results.append({
                "key": doc['key'],
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "total_documents": len(documents),
        "successfully_processed": processed_count,
        "failed": len(documents) - processed_count,
        "results": results
    }


class BucketRequest(BaseModel):
    bucket_name: str

class NumberRequest(BucketRequest):
    mobile_number: str

class BulkNumbersRequest(BucketRequest):
    numbers: list[str]

# Function to fetch numbers from S3
def get_allowed_numbers(bucket_name):
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=FILE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data.get("allowed_numbers", [])
    except s3_client.exceptions.NoSuchKey:
        return []  # If file doesn't exist, return empty list
    except s3_client.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail="Bucket not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

# Function to update S3 file
def update_allowed_numbers(bucket_name, numbers):
    try:
        data = json.dumps({"allowed_numbers": numbers})
        s3_client.put_object(Bucket=bucket_name, Key=FILE_KEY, Body=data, ContentType="application/json")
    except s3_client.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail="Bucket not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating data: {str(e)}")



# FastAPI application
app = FastAPI(title="Document Retrieval System")

# Add this to your FastAPI app

class S3BatchProcessRequest(BaseModel):
    bucket: str
    prefix: Optional[str] = None
    max_documents: Optional[int] = None
    file_extensions: Optional[List[str]] = ['.pdf', '.docx']


@app.post("/s3/process-all")
async def process_all_s3_documents_endpoint(
    request: S3BatchProcessRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start a background job to process all documents in an S3 bucket.
    """
    # Get the list of documents matching criteria
    documents = list_s3_documents(
        request.bucket, 
        request.prefix, 
        request.file_extensions
    )
    
    # Apply max_documents limit if specified
    if request.max_documents is not None:
        documents = documents[:request.max_documents]
    
    # Add background task for each document
    keys = [doc['key'] for doc in documents]
    background_tasks.add_task(
        process_s3_documents_batch, 
        request.bucket, 
        keys, 
        background_tasks
    )
    
    return {
        "message": f"Started background processing of {len(documents)} documents",
        "total_documents": len(documents)
    }

# API Models for S3 operations
class S3ListRequest(BaseModel):
    bucket: str
    prefix: Optional[str] = None
    file_extensions: Optional[List[str]] = None

class S3ProcessRequest(BaseModel):
    bucket: str
    keys: List[str]

class S3EventRequest(BaseModel):
    Records: List[Dict[str, Any]]

# Add these endpoints to your FastAPI app

@app.post("/s3/list")
async def list_s3_documents_endpoint(request: S3ListRequest):
    """List documents available in an S3 bucket."""
    try:
        documents = list_s3_documents(
            request.bucket,
            request.prefix,
            request.file_extensions
        )
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing S3 documents: {str(e)}"
        )

@app.post("/s3/process")
async def process_s3_documents_endpoint(
    request: S3ProcessRequest, 
    background_tasks: BackgroundTasks
):
    """
    Process documents from an S3 bucket and add to the knowledge base.
    Documents are processed in the background.
    """
    try:
        result = await process_s3_documents_batch(
            request.bucket, 
            request.keys,
            background_tasks
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing S3 documents: {str(e)}"
        )


# API endpoints
@app.get("/")
def home():
    """Root endpoint."""
    return {"message": "WhatsApp RAG Webhook Running"}


@app.get("/webhook")
async def webhook_verification(request: Request):
    """
    Handle WhatsApp webhook verification.
    This endpoint is called by WhatsApp to verify the webhook.
    """
    query_params = request.query_params
    mode = query_params.get("hub.mode")
    challenge = query_params.get("hub.challenge")
    token = query_params.get("hub.verify_token")
    
    if mode and token:
        if mode == "subscribe" and token == MY_TOKEN:
            return int(challenge)
        else:
            raise HTTPException(status_code=403, detail="Forbidden")
    
    raise HTTPException(status_code=400, detail="Bad Request")


@app.post("/webhook")
async def webhook_handler(request: Request):
    """
    Handle incoming WhatsApp messages.
    This endpoint receives message notifications from WhatsApp.
    """
    try:
        body_param = await request.json()
        if not body_param.get("object"):
            raise HTTPException(status_code=404, detail="Invalid request")

        try:
            entry = body_param.get("entry", [])
            if not entry:
                print("No entries found in webhook payload.")
                return {"status": "No new entries"}
            entry = entry[0]

            changes = entry.get("changes", [])
            if not changes:
                print("No changes found in webhook payload.")
                return {"status": "No new changes"}
            changes = changes[0]

            value = changes.get("value", {})
            messages = value.get("messages", [])
            if not messages:
                return {"status": "No new messages"}
            
            statuses = value.get("statuses", [])
            phone_number_id = value["metadata"]["phone_number_id"]
            sender = messages[0]["from"]
            message_text = messages[0]["text"]["body"]
            user_name = value.get("contacts", [{}])[0].get("profile", {}).get("name", "User")

            # Check if sender is authorized
            BUCKET = os.getenv("S3_BUCKET_MOBILE", "ep-mobile-numbers")
            ALLOW_MOBILE = get_allowed_numbers(BUCKET)
            print("ALLOW_MOBILE list: ", ALLOW_MOBILE)
            if sender not in ALLOW_MOBILE:
                answer = "Sorry, You are not authorized"
            else:
                # Process the message for authorized users
                query_answer = search_user_query(message_text)
                if query_answer is not None:
                    answer = query_answer
                else:
                    answer = "Sorry, I am not able to answer your query."

            # Send response via WhatsApp API
            url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages?access_token={ACCESS_TOKEN}"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": sender,  # Using the sender's number for reply
                "type": "text",
                "text": {"body": answer}
            }
            headers = {"Content-Type": "application/json"}

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    print("Message sent successfully:", response.json())
                    return {"message": "Message sent successfully"}
                else:
                    print("Error sending message:", response.json())
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Failed to send message",
                            "details": response.json()
                        }
                    )

        except Exception as e:
            print(f"Error processing webhook: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing the webhook: {str(e)}"
            )

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# @app.post("/webhook")
# async def webhook_handler(request: Request):
#     """
#     Handle incoming WhatsApp messages.
#     This endpoint receives message notifications from WhatsApp.
#     """
#     try:
#         body_param = await request.json()
#         if not body_param.get("object"):
#             raise HTTPException(status_code=404, detail="Invalid request")

#         try:
#             entry = body_param.get("entry", [])
#             if not entry:
#                 print("No entries found in webhook payload.")
#                 return {"status": "No new entries"}
#             entry = entry[0]

#             changes = entry.get("changes", [])
#             if not changes:
#               print("No changes found in webhook payload.")
#               return {"status": "No new changes"}
#             changes = changes[0]

#             value = changes.get("value", {})
#             messages = value.get("messages", [])
#             if not messages:
#                 return {"status": "No new messages"}
            
#             statuses = value.get("statuses", [])
#             phone_number_id = value["metadata"]["phone_number_id"]
#             sender = messages[0]["from"]
#             message_text = messages[0]["text"]["body"]
#             user_name = value.get("contacts", [{}])[0].get("profile", {}).get("name", "User")

#             if statuses:
#               # only process status if they exist
#                 statuses = statuses[0]
#                 recipient_id = statuses.get("recipient_id")

#                 if recipient_id in ALLOW_MOBILE:
#                     # Process the message
#                     answer = search_user_query(message_text)
#                     if answer is None:
#                         answer = "Sorry, I am not able to answer your query."
#                 else:
#                     answer = "Sorry, You are not authorized"


#             # Send response via WhatsApp API
#             url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages?access_token={ACCESS_TOKEN}"
#             payload = {
#                 "messaging_product": "whatsapp",
#                 "recipient_type": "individual",
#                 "to": sender,  # Using the sender's number for reply
#                 "type": "text",
#                 "text": {"body": answer}
#             }
#             headers = {"Content-Type": "application/json"}

#             async with httpx.AsyncClient() as client:
#                 response = await client.post(url, json=payload, headers=headers)

#                 if response.status_code == 200:
#                     print("Message sent successfully:", response.json())
#                     return {"message": "Message sent successfully"}
#                 else:
#                     print("Error sending message:", response.json())
#                     raise HTTPException(
#                         status_code=400,
#                         detail={
#                             "error": "Failed to send message",
#                             "details": response.json()
#                         }
#                     )

#         except Exception as e:
#             print(f"Error processing webhook: {str(e)}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"An error occurred while processing the webhook: {str(e)}"
#             )

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"An unexpected error occurred: {str(e)}"
#         )

# @app.post("/webhook")
# async def webhook_handler(request: Request):
#     """
#     Handle incoming WhatsApp messages.
#     This endpoint receives message notifications from WhatsApp.
#     """
#     try:
#         body_param = await request.json()
        
#         if not body_param.get("object"):
#             raise HTTPException(status_code=404, detail="Invalid request")
            
#         try:
#             entry = body_param.get("entry", [])[0]
#             changes = entry.get("changes", [])[0]
#             value = changes.get("value", {})
#             messages = value.get("messages", [])
#             # statuses = value.get("statuses", [])[0]
#             # print(statuses)
#             # recipient_id = statuses[0]["recipient_id"]
#             # print(recipient_id)
            
#             if not messages:
#                 return {"status": "No new messages"}
                
#             phone_number_id = value["metadata"]["phone_number_id"]
#             sender = messages[0]["from"]
#             message_text = messages[0]["text"]["body"]
#             user_name = value.get("contacts", [{}])[0].get("profile", {}).get("name", "User")
            
#             # Process the message
#             answer = search_user_query(message_text)
#             if answer is None:
#                 answer = "Sorry, I am not able to answer your query."
            
#             # Send response via WhatsApp API
#             url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages?access_token={ACCESS_TOKEN}"
#             payload = {
#                 "messaging_product": "whatsapp",
#                 "recipient_type": "individual",
#                 "to": sender,  # Using the sender's number for reply
#                 "type": "text",
#                 "text": {"body": answer}
#             }
#             headers = {"Content-Type": "application/json"}
            
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(url, json=payload, headers=headers)
                
#                 if response.status_code == 200:
#                     print("Message sent successfully:", response.json())
#                     return {"message": "Message sent successfully"}
#                 else:
#                     print("Error sending message:", response.json())
#                     raise HTTPException(
#                         status_code=400,
#                         detail={
#                             "error": "Failed to send message",
#                             "details": response.json()
#                         }
#                     )
                    
#         except Exception as e:
#             print(f"Error processing webhook: {str(e)}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"An error occurred while processing the webhook: {str(e)}"
#             )
            
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"An unexpected error occurred: {str(e)}"
#         )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    Extracts text, chunks it, and stores vectors in ChromaDB.
    """
    temp_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Save the uploaded file temporarily
        content = await file.read()
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Check if the file exists
        if not os.path.exists(temp_path):
            raise HTTPException(
                status_code=500,
                detail=f"File not found: {temp_path}"
            )
        
        # Extract text based on file type
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(
                temp_path, 
                f"{TEMP_DIR}/{os.path.splitext(file.filename)[0]}"
            )
        elif file.filename.lower().endswith(".docx"):
            text = extract_text_from_docx(temp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )
        
        # Store in ChromaDB
        result = store_document_in_chromadb(file.filename, text)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/search")
async def search_endpoint(request: QueryRequest):
    """
    Search the knowledge base with a query.
    Returns a response generated by the LLM based on retrieved content.
    """
    query = request.query
    response_text = search_user_query(query)
    
    if not response_text:
        return {"response": "No relevant data found"}
        
    return {"response": response_text}


@app.post("/numbers/list")
def get_numbers(request: BucketRequest):
    numbers = get_allowed_numbers(request.bucket_name)
    return {"allowed_numbers": numbers}

# 2️⃣ **Add a Single Mobile Number**
@app.post("/numbers/add")
def add_number(request: NumberRequest):
    numbers = get_allowed_numbers(request.bucket_name)
    
    if request.mobile_number in numbers:
        raise HTTPException(status_code=400, detail="Number already exists")
    
    numbers.append(request.mobile_number)
    update_allowed_numbers(request.bucket_name, numbers)
    
    return {"message": "Number added successfully", "added_number": request.mobile_number}

# 3️⃣ **Add Multiple Numbers**
@app.post("/numbers/bulk_add")
def add_bulk_numbers(request: BulkNumbersRequest):
    numbers = get_allowed_numbers(request.bucket_name)
    new_numbers = set(request.numbers) - set(numbers)  # Avoid duplicates

    if not new_numbers:
        raise HTTPException(status_code=400, detail="All numbers already exist")

    numbers.extend(new_numbers)
    update_allowed_numbers(request.bucket_name, numbers)
    
    return {"message": "Numbers added successfully", "added_numbers": list(new_numbers)}

# 4️⃣ **Delete a Single Mobile Number**
@app.post("/numbers/delete")
def delete_number(request: NumberRequest):
    numbers = get_allowed_numbers(request.bucket_name)

    if request.mobile_number not in numbers:
        raise HTTPException(status_code=404, detail="Number not found")

    numbers.remove(request.mobile_number)
    update_allowed_numbers(request.bucket_name, numbers)
    
    return {"message": "Number deleted successfully", "deleted_number": request.mobile_number}

# 5️⃣ **Delete Multiple Numbers**
@app.post("/numbers/bulk_delete")
def delete_bulk_numbers(request: BulkNumbersRequest):
    numbers = get_allowed_numbers(request.bucket_name)
    removed_numbers = [num for num in request.numbers if num in numbers]

    if not removed_numbers:
        raise HTTPException(status_code=404, detail="No numbers found to delete")

    numbers = [num for num in numbers if num not in removed_numbers]
    update_allowed_numbers(request.bucket_name, numbers)

    return {"message": "Numbers deleted successfully", "deleted_numbers": removed_numbers}

# Lambda handler for AWS deployment
# def lambda_handler(event, context):
#     """
#     AWS Lambda entry point.
#     Processes a query from the Lambda event.
#     """
#     try:
#         user_query = event.get("query", "No query provided")
#         response_message = search_user_query(user_query)
        
#         return {
#             "statusCode": 200,
#             "body": json.dumps({"response": response_message})
#         }
#     except Exception as e:
#         return {
#             "statusCode": 500,
#             "body": json.dumps({"error": str(e)})
#         }


# # Main entry point for direct execution
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)