import boto3
import json

prompt_data = """
Act as a shakesphere and write a poem on Machine Learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt":"<s>[INST]" +prompt_data+ "[/INST]",
    "max_tokens":200,
    "temperature":0.5,
    "top_p":0.9,
    "top_k":50
}

body = json.dumps(payload)
model_id="mistral.mistral-7b-instruct-v0:2"
respone=bedrock.invoke_model(
    modelId=model_id,
    contentType= "application/json",
    accept= "application/json",
    body=body
)

respone_body = json.loads(respone.get("body").read())
print(respone_body['outputs'][0]['text'])