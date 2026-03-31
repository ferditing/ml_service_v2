# animal_normalizer.py
import re

ANIMAL_MAP = {
    "cow": ["cow", "cattle", "bull", "calf", "heifer"],
    "goat": ["goat", "kid"],
    "sheep": ["sheep", "ram", "ewe", "lamb"],
    "poultry": ["poultry", "chicken", "hen", "rooster", "duck", "turkey"]
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


def detect_animal_from_text(text: str) -> str | None:
    """
    Try to detect animal type mentions in free text.
    Returns canonical animal type or None.
    """
    normalized = normalize_text(text)
    
    # Check if any animal type is mentioned in the text
    for canonical, variants in ANIMAL_MAP.items():
        # Check canonical name
        if f" {canonical} " in f" {normalized} ":
            return canonical
        # Check variants
        for variant in variants:
            if f" {variant} " in f" {normalized} ":
                return canonical
    
    return None
