import json
from config import bedrock_runtime

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
        "max_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50
    }

    body = json.dumps(payload)
    model_id = "mistral.mistral-7b-instruct-v0:2"
    respone = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    response_body = json.loads(respone.get("body").read())
    return response_body
