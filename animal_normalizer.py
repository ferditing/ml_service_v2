# animal_normalizer.py
import re

ANIMAL_MAP = {
    "cow": ["cow", "cattle", "bull", "calf", "heifer"],
    "goat": ["goat", "kid"],
    "sheep": ["sheep", "ram", "ewe", "lamb"]
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_to_canonical_animal(animal_input: str) -> str | None:
    normalized = normalize_text(animal_input)

    for canonical, variants in ANIMAL_MAP.items():
        if normalized == canonical or normalized in variants:
            return canonical

    return None
