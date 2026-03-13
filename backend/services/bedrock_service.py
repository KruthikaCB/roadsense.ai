import boto3
import json
from config import settings

bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name=settings.AWS_REGION
)

def generate_complaint(latitude: float, longitude: float, severity: str, image_url: str) -> str:
    prompt = f"""Human: Generate a formal BBMP road complaint letter for the following pothole:
    Location coordinates: {latitude}, {longitude}
    Severity level: {severity}
    Evidence: {image_url}
    
    Write a short professional 4-5 line complaint requesting immediate repair.
    
Assistant:"""

    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.5
    })

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["completion"]
    except Exception as e:
        raise Exception(f"Bedrock failed: {str(e)}")
