# nlp_service.py

import joblib
import numpy as np
import pandas as pd

from fuzzy_matcher import match_symptoms
from animal_normalizer import map_to_canonical_animal
from symptom_map import SYMPTOM_MAP

MODEL_PATH = "decision_tree_model.pkl"

# Load model once
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]
LABELS = artifact["label_encoder_classes"]

def predict_from_nlp(
    animal_input: str,
    symptom_text: str,
    age: float | None = None,
    body_temperature: float | None = None,
):
    # ---------------------------
    # 1. Normalize animal
    # ---------------------------
    animal = map_to_canonical_animal(animal_input)
    if not animal:
        return {
            "error": "Unsupported or unknown animal type",
            "animal_input": animal_input
        }

    # ---------------------------
    # 2. Extract symptoms (fuzzy NLP)
    # ---------------------------
    nlp_result = match_symptoms(symptom_text)
    matched_symptoms = nlp_result["matched_symptoms"]

    if not matched_symptoms:
        return {
            "error": "No recognizable symptoms found",
            "symptom_text": symptom_text
        }

    # ---------------------------
    # 3. Build feature vector
    # ---------------------------
    features = dict.fromkeys(SYMPTOM_MAP, 0)

    # numeric features
    features["age"] = age if age is not None else 0
    features["body_temperature"] = body_temperature if body_temperature is not None else 0

    # animal one-hot
    features[f"animal_{animal}"] = 1

    # symptom flags
    for symptom in matched_symptoms:
        if symptom in features:
            features[symptom] = 1

    # ---------------------------
    # 4. Predict
    # ---------------------------
    X = pd.DataFrame(
        [[features.get(f, 0) for f in FEATURES]],
        columns=FEATURES
    )
    pred_idx = int(model.predict(X)[0])
    predicted_disease = LABELS[pred_idx]
    probabilities = model.predict_proba(X)[0]

    confidence = float(max(probabilities))

    # ---------------------------
    # 5. Response
    # ---------------------------
    return {
        "animal": animal,
        "predicted_disease": predicted_disease,
        "confidence": round(confidence, 3),
        "matched_symptoms": nlp_result,
        "used_features": [k for k, v in features.items() if v == 1]
    }
