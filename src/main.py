# from bedrock_client import ask_bedrock
# from vector_store import store_documents,search_knowledge_base
#
# # Sample knowledge base
# documents = [
#     {"id": "1", "text": "AWS Lambda is a serverless computing service from Amazon Web Services."},
#     {"id": "2", "text": "Amazon S3 is an object storage service that offers industry-leading scalability."},
#     {"id": "3", "text": "Amazon Bedrock allows developers to build and scale AI models securely on AWS."},
# ]
#
# # Store documents in vector database
# store_documents(documents)
#
# # User query
# query = "What is Amazon Bedrock?"
# retrieved_text = search_knowledge_base(query)
#
# # Get AI-generated response
# if retrieved_text:
#     response = ask_bedrock(query, retrieved_text)
#     print("\nAmazon Bedrock Response:", response)
# else:
#     print("No relevant data found.")

from vector_store import store_documents_from_word, store_documents_from_pdf, search_knowledge_base
from bedrock_client import ask_bedrock

def store_by_document_type():
    doc_path = "../docs/"
    pdf_path = "../pdfs/"
    doc_type = input("Enter Document type PDF or Word? [P/W]:")
    if doc_type.lower() == 'w' or doc_type.lower() == 'word':
        store_documents_from_word(doc_path)
    elif doc_type.lower() == 'p' or doc_type.lower() == 'pdf':
        store_documents_from_pdf(pdf_path)
    else:
        try_again = input("Invalid input. Do you want to try again?[Y/N]:")
        if try_again.lower() == 'y' or try_again.lower() == 'yes':
            store_by_document_type()
        else:
            return

    more_doc = input("Do you want to store more documents? [Y/N]:")
    if more_doc.lower() == 'y' or more_doc.lower() == 'yes':
        store_by_document_type()

def store_documents():
    # Step 1: Store documents in ChromaDB
    store_doc = input("Do you want to store documents? [Y/N]:")
    if store_doc.lower() == 'y' or store_doc.lower() == 'yes':
        store_by_document_type()

def search_user_query():
    #Query Examples:
    # query = "What is the setup cost for this mandate?"
    # query = "What is name of client and his pan number and his address, also on which date this mandate was signed?"
    # query1 = "how many Required fields that will be fetched from Caliber during integration of non functional requirements?"
    # query2 = "How to set up family trust?"
    while True:
        query = input("\nEnter your query:").strip()
        if query == "-1":
            break
        retrieved_text = search_knowledge_base(query)
        if retrieved_text:
            response = ask_bedrock(query, retrieved_text)
            print(f"\n\033[34m{response['outputs'][0]['text']}\033[0m")
        else:
            print("No relevant data found.")


def main():
    store_documents()
    search_user_query()

if __name__=="__main__":
    main()