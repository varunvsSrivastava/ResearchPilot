from openai import OpenAI

from researchpilot.config import (
    get_openai_api_key,
    get_openai_model
)

def create_client() -> str:
    return OpenAI(api_key=get_openai_api_key())

def generate_response(prompt:str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    
    client = create_client()
    
    response = client.responses.create(
        model = get_openai_model(),
        input = prompt
    )
    
    return response.output_text