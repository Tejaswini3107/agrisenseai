import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="bharatgenai/AgriParam",
    token=os.getenv("HF_TOKEN")
)


def ask_agriparam(question: str, context: str = None) -> str:
    if context:
        prompt = f"<context> {context} <user> {question} <assistant>"
    else:
        prompt = f"<user> {question} <assistant>"

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        return response.strip()
    except Exception as e:
        return f"AgriParam service unavailable: {str(e)}"
    
