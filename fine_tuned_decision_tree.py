import re
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

OUT_PATH = "fine_tuned_decision_tree_model.pkl"

ROOT = os.path.dirname(__file__)
OUT_PATH = os.path.join(ROOT, "decision_tree_model.pkl")  # same name ml_service expects


def normalize_symptom(s):
    if pd.isna(s): return "none"
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().replace(' ', '_')

def symptom_col_name(s):
    return s.replace(' ', '_').replace('-', '_')


def main():
    df = pd.read_csv('animal_data.csv')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    for col in ["Symptom 1", "Symptom 2", "Symptom 3"]:
        df[col] = df[col].apply(normalize_symptom)

    # Filter to valid animals BEFORE creating species lock feature
    valid_animals = ['cow', 'goat', 'sheep', 'poultry']
    df = df[df['Animal'].str.lower().isin(valid_animals)]

# 6. Animal-Symptom Interaction (The Species Lock)
    df['Animal_Symptom_1'] = df['Animal'].str.lower() + "_" + df['Symptom 1']

    # 7. Identify all unique symptoms for Binary Expansion
    symptom_cols = ["Symptom 1", "Symptom 2", "Symptom 3"]
    symptom_set = set()
    for col in symptom_cols:
        symptom_set.update(df[col].dropna().unique())
    symptoms = sorted(list(symptom_set))

    # 8. Binary Expansion (Manual One-Hot)
    for s in symptoms:
        col = symptom_col_name(s)
        df[col] = df[symptom_cols].apply(lambda row: int(s in row.values), axis=1)

    # --- BUILD ANIMAL-SYMPTOM MAPPING FOR FRONTEND FILTERING ---
    # Do this BEFORE one-hot encoding so we still have 'Animal' column
    animal_symptoms_map = {}
    for animal in valid_animals:
        animal_rows = df[df['Animal'].str.lower() == animal]
        animal_symptoms_set = set()
        for symptom_col in symptom_cols:
            if symptom_col in animal_rows.columns:
                animal_symptoms_set.update(animal_rows[symptom_col].dropna().unique())
        animal_symptoms_map[animal] = sorted(list(animal_symptoms_set))

    # 9. One-Hot Encode Animal Type
    df = pd.get_dummies(df, columns=['Animal'], prefix='animal')

    # 10. One-Hot Encode Species-Lock Feature
    df = pd.get_dummies(df, columns=['Animal_Symptom_1'], prefix='interact')

    # 11. Final Cleanup (Rename)
    rename_map = {"Age": "age", "Temperature": "body_temperature", "Disease": "disease"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Drop raw text symptoms
    df = df.drop(columns=["Symptom 1", "Symptom 2", "Symptom 3"], errors='ignore')

    df = df.dropna(subset=['disease']).fillna(0)
    print(f"Final Dataset Ready. Shape: {df.shape}")
    print(f"Feature Engineering Complete. Total columns: {len(df.columns)}")

    # --- Extract symptom names for audit ---

    # --- Cell 6: Final Training, Evaluation & Vocabulary Audit ---

    # 1. Label Encoding Target
    le = LabelEncoder()
    y = le.fit_transform(df['disease'].astype(str))
    label_classes = list(le.classes_)

    # 2. Define Features
    X = df.drop(columns=['disease'], errors='ignore')
    feature_names = list(X.columns)

    # 3. Train/Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. SMOTE to balance the training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # 6. Evaluate
    y_pred = clf.predict(X_test)

    print("\n" + "="*30)
    print("--- PERFORMANCE REPORT ---")
    print("="*30)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # 7. PRINT UNIQUE DISEASES
    print("\n" + "="*30)
    print(f"UNIQUE DISEASES ({len(label_classes)})")
    print("="*30)
    for i, disease in enumerate(sorted(label_classes), 1):
        print(f"{i}. {disease}")

    # 8. PRINT UNIQUE SYMPTOMS
    # We extract these from the feature names. 
    # We exclude 'age', 'body_temperature', and the 'animal_' or 'interact_' prefixes to get the raw symptoms.
    print("\n" + "="*30)
    unique_symptoms = sorted([
        f for f in feature_names 
        if f not in ['age', 'body_temperature'] 
        and not f.startswith('animal_') 
        and not f.startswith('interact_')
    ])
    print(f"UNIQUE SYMPTOMS ({len(unique_symptoms)})")
    print("="*30)
    # Print in columns for readability
    for i, symptom in enumerate(unique_symptoms, 1):
        print(f"{i:2}. {symptom:<25}", end="\n" if i % 3 == 0 else "")

    # 9. Save Artifact
    print(f"\nAnimal-Symptom Mapping (for filtering):")
    for animal, symptoms in animal_symptoms_map.items():
        print(f"  {animal}: {len(symptoms)} symptoms")
    
    artifact = {
        "model": clf,
        "features": feature_names,
        "label_encoder_classes": label_classes,
        "animal_symptoms": animal_symptoms_map  # Mapping for frontend filtering
    }
    joblib.dump(artifact, OUT_PATH)
    print(f"\n\nModel saved to {OUT_PATH}")


if __name__ == "__main__":
    main()