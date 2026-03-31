from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from nlp_service import predict_from_nlp
from fuzzy_matcher import match_symptoms
from animal_normalizer import map_to_canonical_animal, detect_animal_from_text

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "decision_tree_model.pkl")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]
LABELS = artifact["label_encoder_classes"]
ANIMAL_SYMPTOMS = artifact.get("animal_symptoms", {})  # Load animal-symptom mapping

app = FastAPI(title="Smart Livestock ML Service v2")


@app.get("/health")
def health():
    # Filter to show only displayable symptoms (exclude internal features)
    displayable_symptoms = [
        f for f in FEATURES 
        if f not in ['age', 'body_temperature'] 
        and not f.startswith('animal_') 
        and not f.startswith('interact_')
    ]
    return {
        "status": "ok", 
        "features": displayable_symptoms,  # For frontend compatibility
        "symptoms": displayable_symptoms,  # Explicit symptoms field
        "labels": LABELS,
        "all_features": FEATURES  # For debugging only
    }


@app.post("/predict")
def predict(payload: dict):
    if any(k in payload for k in ("symptom_text", "symptoms_text", "symptoms", "raw_symptoms")):
        raise HTTPException(
            status_code=400,
            detail=(
                "This endpoint does not process raw symptom text. "
                "Use the NLP service (nlp_service.py) to extract features from free text."
            ),
        )

    try:
        row = [payload.get(f, 0) for f in FEATURES]
        X = pd.DataFrame([row], columns=FEATURES)
        pred_idx = int(model.predict(X)[0])
        proba = float(model.predict_proba(X).max())

        return {"predicted_label": LABELS[pred_idx], "confidence": round(proba, 3)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_from_text")
def predict_from_text(payload: dict):
    # Delegate free-text processing to the NLP helper in nlp_service.py
    animal = payload.get("animal")
    symptom_text = payload.get("symptom_text")
    if not animal or not symptom_text:
        raise HTTPException(status_code=400, detail="`animal` and `symptom_text` required")

    # Forward to the NLP function which will build features and call the model.
    return predict_from_nlp(animal, symptom_text, payload.get("age"), payload.get("body_temperature"))


@app.post("/normalize")
def normalize(payload: dict):
    """
    Normalize/extract canonical animal type and symptoms from user input.
    Useful for preprocessing reports before storage.
    Can detect animal type from symptom text if not explicitly provided.
    """
    animal_input = payload.get("animal", "").strip()
    symptom_text = payload.get("symptom_text", "").strip()
    
    if not symptom_text:
        raise HTTPException(status_code=400, detail="`symptom_text` required")
    
    try:
        # Normalize animal type: first try explicit input, then detect from text
        canonical_animal = None
        if animal_input:
            canonical_animal = map_to_canonical_animal(animal_input)
        
        # If no animal from input, try to detect from symptom text
        if not canonical_animal:
            canonical_animal = detect_animal_from_text(symptom_text)
        
        # Extract canonical symptoms using fuzzy matching
        symptom_result = match_symptoms(symptom_text, score_threshold=65)
        matched_symptoms = symptom_result.get("matched_symptoms", [])
        confidence = symptom_result.get("confidence", 0)
        
        return {
            "animal_type": canonical_animal,
            "matched_symptoms": matched_symptoms,
            "symptom_confidence": confidence,
            "symptom_text": symptom_text,
            "success": bool(matched_symptoms)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symptoms_for_animal/{animal}")
def symptoms_for_animal(animal: str):
    """
    Get symptoms that are valid/occur for a specific animal type.
    Uses the animal-symptom mapping extracted from training data.
    """
    animal_lower = animal.lower().strip()
    
    # Validate animal
    canonical_animal = map_to_canonical_animal(animal_lower)
    if not canonical_animal:
        raise HTTPException(status_code=400, detail=f"Unknown animal type: {animal}")
    
    # Get symptoms from the pre-built mapping
    valid_symptoms = ANIMAL_SYMPTOMS.get(canonical_animal, [])
    
    # If mapping is empty, return all base symptoms as fallback
    if not valid_symptoms:
        valid_symptoms = [
            f for f in FEATURES 
            if f not in ['age', 'body_temperature'] 
            and not f.startswith('animal_') 
            and not f.startswith('interact_')
        ]
    
    return {
        "animal": canonical_animal,
        "symptoms": valid_symptoms,
        "count": len(valid_symptoms)
    }


