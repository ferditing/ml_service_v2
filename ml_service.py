from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from nlp_service import predict_from_nlp

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "decision_tree_model.pkl")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]
LABELS = artifact["label_encoder_classes"]

app = FastAPI(title="Smart Livestock ML Service v2")


@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES, "labels": LABELS}


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
