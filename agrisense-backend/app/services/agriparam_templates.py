"""AgriParam prompt templates with DB-backed persistence.

Templates are stored per language in `agriparam_templates` and fall back to
generated defaults for all supported AgriParam languages.
"""

from typing import Any, Dict, cast

from sqlalchemy.orm import Session

from app.database import AgriParamTemplate
from app.services.agriparam import SUPPORTED_LANGUAGES

LANGUAGE_ALIASES = {
    "en": "english",
    "hi": "hindi",
    "as": "assamese",
    "bn": "bengali",
    "brx": "bodo",
    "doi": "dogri",
    "gu": "gujarati",
    "kn": "kannada",
    "ks": "kashmiri",
    "kok": "konkani",
    "mai": "maithili",
    "ml": "malayalam",
    "mni": "manipuri",
    "mr": "marathi",
    "ne": "nepali",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "sat": "santali",
    "sd": "sindhi",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
    "ar": "arabic",
    "fr": "french",
}


def normalize_language_key(language: str | None) -> str:
    value = (language or "english").strip().lower()
    if value in SUPPORTED_LANGUAGES:
        return value
    if value in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[value]
    return "english"


def _default_template(language_key: str) -> str:
    language_label = SUPPORTED_LANGUAGES.get(language_key, "English")
    return (
        f"You are an agricultural advisor. Answer in {language_label}.\n"
        "Farmer context:\n"
        "Soil: {soil}\n"
        "Climate: {climate}\n"
        "Available water (mm): {water}\n"
        "Goal: {goals}\n"
        "Return:\n"
        "1) Recommended crop\n"
        "2) Brief reason\n"
        "3) Three practical next steps"
    )


def default_templates() -> Dict[str, str]:
    return {language: _default_template(language) for language in SUPPORTED_LANGUAGES.keys()}


def get_templates_map(db: Session) -> Dict[str, str]:
    templates = default_templates()
    rows = db.query(AgriParamTemplate).all()
    for row in rows:
        row_obj = cast(Any, row)
        language = normalize_language_key(getattr(row_obj, "language", None))
        template_text = getattr(row_obj, "template_text", None)
        if isinstance(template_text, str) and template_text:
            templates[language] = template_text
    return templates


def upsert_templates(db: Session, templates: Dict[str, str]) -> Dict[str, str]:
    if not templates:
        return get_templates_map(db)

    for language_raw, template_text in templates.items():
        language = normalize_language_key(language_raw)
        content = (template_text or "").strip()
        if not content:
            continue

        existing = db.query(AgriParamTemplate).filter(AgriParamTemplate.language == language).first()
        if existing:
            existing_obj = cast(Any, existing)
            existing_obj.template_text = content
        else:
            db.add(AgriParamTemplate(language=language, template_text=content))

    db.commit()
    return get_templates_map(db)


def get_template(lang: str, db: Session | None = None) -> str:
    language = normalize_language_key(lang)
    if db is None:
        return default_templates()[language]

    templates = get_templates_map(db)
    return templates.get(language, templates["english"])


def format_prompt(lang: str, db: Session | None = None, **kwargs: Any) -> str:
    """Format the template for the language with provided kwargs.

    Example: format_prompt('hindi', db=db, soil='loam', climate='tropical', water=120, goals='yield')
    """
    template = get_template(lang, db=db)
    try:
        return template.format(**kwargs)
    except Exception:
        return template
