import re
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, "animal_disease_dataset.csv")
OUT_PATH = os.path.join(ROOT, "decision_tree_model.pkl")  # same name ml_service expects

def normalize_symptom(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def symptom_col_name(s):
    return s.replace(' ', '_').replace('-', '_')

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV:", CSV_PATH, "shape:", df.shape)
    
    # Build symptom vovubulary (Handle missing data gracefully)
    symptom_cols = [c for c in ["Symptom 1", "Symptom 2", "Symptom 3"] if c in df.columns]
    symptom_set = set()
    for col in symptom_cols:
        symptom_set.update(df[col].dropna().astype(str).map(normalize_symptom).unique())
    symptom_set.discard(None)
    symptoms = sorted(symptom_set)
    print("Found", len(symptoms), "unique symptom phrases")

    #  Expand symptoms into binary collumns
    for s in symptoms:
        col = symptom_col_name(s)
        def has_symptom(r, s=s):
            try:
                return int(
                    (normalize_symptom(r.get("Symptom 1")) == s if "Symptom 1"  in r else False)or
                    (normalize_symptom(r.get("Symptom 2")) == s if "Symptom 2"  in r else False)or
                    (normalize_symptom(r.get("Symptom 3")) == s if "Symptom 3"  in r else False)
                )
            except Exception:
                return 0
        df[col] = df.apply(has_symptom, axis=1)
    print("Sample of binary symptom columns:")
    print(df[symptom_col_name(symptoms[1])].head())

    #  Drop original columns if still present 
    for c in ["Symptom 1", "Symptom 2", "Symptom 3"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Rename columns if present
    rename_map = {}
    if "Animal" in df.columns: rename_map["Animal"] = "animal_type"
    if "Age" in df.columns: rename_map["Age"] = "age"
    if "Temperature" in df.columns: rename_map["Temperature"] = "body_temperature"
    if "Disease" in df.columns: rename_map["Disease"] = "disease"
    if rename_map:
        df = df.rename(columns=rename_map)
    

    #  Drop buffallo columns
    if 'animal_type' in df.columns:
        valid_animals = ['cow', 'goat', 'sheep']
        df = df[df['animal_type'].str.lower().isin(valid_animals)]
        print("Filtered to valid animal types, new shape:", df.shape[0])
        if 'buffalo' in df['animal_type'].values:
            print("Warning: 'buffalo' entries were found")

    unique_diseases = df['disease'].nunique()
    print("Unique diseases to predict:", unique_diseases)

    if unique_diseases < 2:
        raise ValueError("Not enough unique diseases to train a model.")
    
    # Numeric conversions
    if "age" in df.columns:
        df["age"]=pd.to_numeric(df["age"], errors='coerce')
    if "body_temperature" in df.columns:
        df["body_temperature"]=pd.to_numeric(df["body_temperature"], errors='coerce')
 
    #  Ensure target exists
    if 'disease' not in df.columns:
        raise ValueError("Target column 'disease' not found in dataset.")
    df = df.dropna(subset=['disease'])
    print("After cleaning shape:", df.shape)

    # one hot endcode animal_type if present
    if 'animal_type' in df.columns:
        df = pd.get_dummies(df, columns=['animal_type'], prefix='animal')

    # Label encode target
    le = LabelEncoder()
    df['disease_label'] = le.fit_transform(df["disease"].astype(str))
    label_classes = list(le.classes_)
    print("Label classes:", label_classes)

    # Feature columns (exclude raw disease and disease_label)
    feature_cols = [c for c in df.columns if c not in ['disease', 'disease_label']]
    X = df[feature_cols].fillna(0)
    y = df['disease_label']

    print("Number of features:", len(feature_cols))

    #  Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate 
    y_pred = clf.predict(X_test)
    print("Accuracy on holdout:", accuracy_score(y_test, y_pred))
    print("Clasification report:\n", classification_report(y_test, y_pred))

    # save artifact
    artifact = {
        "model": clf,
        "features": list(X.columns),
        "label_encoder_classes": label_classes
    }

    # Backup any existing model
    if os.path.exists(OUT_PATH):
        os.replace(OUT_PATH, OUT_PATH + ".bak")
        print("Backed up existing model to", OUT_PATH + ".bak")

    joblib.dump(artifact, OUT_PATH)
    print("Saved model artifact to", OUT_PATH)
    print("Features saved:", artifact['features'])

if __name__ == "__main__":
    main()