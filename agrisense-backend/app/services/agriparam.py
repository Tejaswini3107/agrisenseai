import os
from typing import Optional
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="bharatgenai/AgriParam",
    token=os.getenv("HF_TOKEN")
)


SUPPORTED_LANGUAGES = {
    "english": "English",
    "hindi": "Hindi",
    "assamese": "Assamese",
    "bengali": "Bengali",
    "bodo": "Bodo",
    "dogri": "Dogri",
    "gujarati": "Gujarati",
    "kannada": "Kannada",
    "kashmiri": "Kashmiri",
    "konkani": "Konkani",
    "maithili": "Maithili",
    "malayalam": "Malayalam",
    "manipuri": "Manipuri",
    "marathi": "Marathi",
    "nepali": "Nepali",
    "odia": "Odia",
    "punjabi": "Punjabi",
    "sanskrit": "Sanskrit",
    "santali": "Santali",
    "sindhi": "Sindhi",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "urdu": "Urdu",
    "arabic": "Arabic",
    "french": "French",
}


def _build_language_instruction(language: str | None) -> str:
    normalized = (language or "english").strip().lower()
    label = SUPPORTED_LANGUAGES.get(normalized, "English")

    if label == "English":
        return "Reply in easy-to-understand English. Keep the answer user-friendly, crisp and concise."

    return (
        f"Reply in {label}. Keep the answer user-friendly, crisp and concise. "
        "If a precise translation is hard, answer in a mix of the requested language and simple English."
    )


def ask_agriparam(question: str, context: Optional[str] = None, language: Optional[str] = "english") -> str:
    language_instruction = _build_language_instruction(language)
    if context:
        prompt = f"{language_instruction}\n<context> {context} <user> {question} <assistant>"
    else:
        prompt = f"{language_instruction}\n<user> {question} <assistant>"

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
