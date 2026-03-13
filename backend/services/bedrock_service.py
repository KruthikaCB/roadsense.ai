import boto3
import json
from config import settings

bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name=settings.AWS_REGION
)

def generate_complaint(latitude: str, longitude: str, severity: str, image_url: str) -> str:
    prompt = f"""Generate a formal BBMP road complaint letter for the following pothole:
    Location coordinates: {latitude}, {longitude}
    Severity level: {severity}
    Evidence photo: {image_url}
    
    Write a short professional 4-5 line complaint requesting immediate repair."""

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.5
        }
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"]
    except Exception as e:
        raise Exception(f"Bedrock failed: {str(e)}")
