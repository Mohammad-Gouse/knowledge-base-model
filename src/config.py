import boto3
import chromadb

# Initialize ChromaDB (Persistent storage)
CHROMA_DB_PATH = "../embeddings"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="knowledge_base")

# Initialize Amazon Bedrock Client
AWS_REGION = "ap-south-1"
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
