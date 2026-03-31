import os

# make sure the model artifact exists (chooses same path as training script)
from train_decision_tree import OUT_PATH, main as train_model

if not os.path.exists(OUT_PATH):
    print("Model artifact missing, running training...")
    train_model()

from nlp_service import predict_from_nlp

result = predict_from_nlp(
    animal_input="cow",
    symptom_text="the animal is weak, not eating and has blisters on mout",
    age=3,
    body_temperature=39.5
)

print(result)
