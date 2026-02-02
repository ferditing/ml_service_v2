# fuzzy_matcher.py
from rapidfuzz import fuzz
from symptom_map import SYMPTOM_MAP
import re

# Flatten all known phrases → canonical symptom
PHRASE_TO_CANONICAL = {}

for canonical, phrases in SYMPTOM_MAP.items():
    for p in phrases:
        PHRASE_TO_CANONICAL[p.lower()] = canonical

KNOWN_PHRASES = list(PHRASE_TO_CANONICAL.keys())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fuzzy_match_phrase(
    phrase: str,
    score_threshold: int = 80
):
    """
    Match a phrase to the closest known symptom phrase.
    Returns (canonical_symptom, score) or (None, score)
    """
    phrase = normalize_text(phrase)

    best_score = 0
    best_match = None

    for known in KNOWN_PHRASES:
        score = fuzz.partial_ratio(phrase, known)
        if score > best_score:
            best_score = score
            best_match = known

    if best_score >= score_threshold:
        return PHRASE_TO_CANONICAL[best_match], best_score

    return None, best_score


def match_symptoms(
    inputs,
    score_threshold: int = 80
):
    """
    inputs: list[str] OR free-text string
    """
    if isinstance(inputs, str):
        # split free text roughly
        phrases = re.split(r',|and|\.|\n', inputs)
    else:
        phrases = inputs

    matched = set()
    unmatched = []
    scores = []

    for p in phrases:
        p = p.strip()
        if not p:
            continue

        canonical, score = fuzzy_match_phrase(p, score_threshold)
        if canonical:
            matched.add(canonical)
            scores.append(score)
        else:
            unmatched.append(p)

    avg_confidence = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "matched_symptoms": list(matched),
        "confidence": avg_confidence,
        "unmatched_phrases": unmatched
    }
