from nlp_service import predict_from_nlp

result = predict_from_nlp(
    animal_input="cow",
    symptom_text="the animal is weak, not eating and has blisters on mout",
    age=3,
    body_temperature=39.5
)

print(result)
