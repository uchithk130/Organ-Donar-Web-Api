from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

# ========================================================
# 1. Load Data & Train Model if Needed
# ========================================================

blood_compatibility = {
    'O-': ['O-'],
    'O+': ['O-', 'O+'],
    'A-': ['A-', 'O-'],
    'A+': ['A-', 'A+', 'O-', 'O+'],
    'B-': ['B-', 'O-'],
    'B+': ['B-', 'B+', 'O-', 'O+'],
    'AB-': ['A-', 'B-', 'AB-'],
    'AB+': ['O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+']
}

def load_data():
    donors_csv = "donors_raw.csv"
    recipients_csv = "recipients_raw.csv"
    merged_csv = "merged_data.csv"
    
    if not Path(merged_csv).exists():
        raise HTTPException(status_code=404, detail=f"{merged_csv} not found.")
    
    return pd.read_csv(merged_csv)

# def train_model():
#     data = load_data()
    
#     # Encoding
#     blood_encoder = LabelEncoder()
#     organ_encoder = LabelEncoder()
    
#     data['D_blood_group'] = blood_encoder.fit_transform(data['D_blood_group'])
#     data['R_blood_group'] = blood_encoder.transform(data['R_blood_group'])
#     data['D_organ'] = organ_encoder.fit_transform(data['D_organ'])
#     data['R_organ'] = organ_encoder.transform(data['R_organ'])
    
#     # Define features
#     donor_features = ['D_age', 'D_weight', 'D_height', 'D_bmi', 'D_blood_group', 'D_organ',
#                       'D_creatinine', 'D_bilirubin', 'D_ejection_fraction', 'D_map',
#                       'D_cholesterol', 'D_glucose_level', 'D_ocular_pressure']
#     recipient_features = ['R_age', 'R_weight', 'R_height', 'R_bmi', 'R_blood_group', 'R_organ',
#                           'R_creatinine', 'R_bilirubin', 'R_ejection_fraction', 'R_map', 'R_urgency',
#                           'R_cholesterol', 'R_glucose_level', 'R_ocular_pressure']
    
#     X = data[donor_features + recipient_features].values
#     y = data['match'].values
    
#     # Split & Scale
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # Train Model
#     model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
#                                    class_weight='balanced', random_state=42)
#     model.fit(X_train, y_train)
    
#     # Save Artifacts
#     joblib.dump(model, 'transplant_rf_model.joblib')
#     joblib.dump(scaler, 'transplant_scaler.joblib')
#     joblib.dump(blood_encoder, 'blood_group_encoder.joblib')
#     joblib.dump(organ_encoder, 'organ_encoder.joblib')
    
#     return model, scaler, blood_encoder, organ_encoder

def load_model():
    try:
        model = joblib.load('transplant_rf_model.joblib')
        scaler = joblib.load('transplant_scaler.joblib')
        blood_encoder = joblib.load('blood_group_encoder.joblib')
        organ_encoder = joblib.load('organ_encoder.joblib')
    except Exception:
        print("Training model as no existing model was found.")
        # return train_model()
    return model, scaler, blood_encoder, organ_encoder

# Load model at startup
model, scaler, blood_encoder, organ_encoder = load_model()

# ========================================================
# 2. API Endpoints
# ========================================================

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/match-donors")
def match_donors(recipient: dict):
    try:
        # Load donor data
        donors = pd.read_csv("donors_raw.csv")
        
        # Blood compatibility filtering
        target_blood = recipient['blood_group']
        compatible_blood = blood_compatibility.get(target_blood, [])

# Special handling for AB+
        if target_blood == 'AB+':
            compatible_blood = blood_compatibility['AB+']

        compatible_donors = donors[
            (donors['organ'] == recipient['organ_needed']) &
            (donors['blood_group'].isin(compatible_blood))
        ]
        
        if compatible_donors.empty:
            return {"message": "No compatible donors found"}
        
        # Process recipient data
        recipient['bmi'] = recipient['weight'] / (recipient['height'] ** 2)
        recipient['map'] = (recipient['bp_systolic'] + 2 * recipient['bp_diastolic']) / 3
        
        if recipient['blood_group'] not in blood_encoder.classes_:
            raise HTTPException(status_code=400, detail="Invalid blood group")
        
        recipient['R_blood_group'] = blood_encoder.transform([recipient['blood_group']])[0]
        recipient['R_organ'] = organ_encoder.transform([recipient['organ_needed']])[0]
        
        # Prepare for prediction
        predictions = []
        for _, donor in compatible_donors.iterrows():
            donor_data = donor.to_dict()
            
            # Encode donor features
            donor_blood = blood_encoder.transform([donor_data['blood_group']])[0]
            donor_organ = organ_encoder.transform([donor_data['organ']])[0]
            
            X_input = np.array([[
                donor_data['age'],           # D_age
                donor_data['weight'],        # D_weight
                donor_data['height'],        # D_height
                donor_data['weight']/(donor_data['height']**2),  # D_bmi
                donor_blood,                 # D_blood_group (encoded)
                donor_organ,                 # D_organ (encoded)
                donor_data['creatinine'],    # D_creatinine
                donor_data['bilirubin'],     # D_bilirubin
                donor_data['ejection_fraction'],  # D_ejection_fraction
                (donor_data['bp_systolic'] + 2*donor_data['bp_diastolic'])/3,  # D_map
                donor_data['cholesterol'],   # D_cholesterol
                donor_data['glucose_level'], # D_glucose_level
                donor_data['ocular_pressure'],  # D_ocular_pressure
                # Recipient features
                recipient['age'],           # R_age
                recipient['weight'],        # R_weight
                recipient['height'],        # R_height
                recipient['bmi'],           # R_bmi
                recipient['R_blood_group'], # R_blood_group (encoded)
                recipient['R_organ'],       # R_organ (encoded)
                recipient['creatinine'],    # R_creatinine
                recipient['bilirubin'],     # R_bilirubin
                recipient['ejection_fraction'],  # R_ejection_fraction
                recipient['map'],           # R_map
                recipient['urgency'],       # R_urgency
                recipient['cholesterol'],   # R_cholesterol
                recipient['glucose_level'], # R_glucose_level
                recipient['ocular_pressure']  # R_ocular_pressure
            ]])
            
            X_input = scaler.transform(X_input)
            proba = model.predict_proba(X_input)[0][1]
            
            predictions.append({
                "donor_id": int(donor.name),
                "name": donor_data['name'],
                "age": int(donor_data['age']),
                "weight": float(donor_data['weight']),
                "height": float(donor_data['height']),
                "blood_group": donor_data['blood_group'],
                "hla": donor_data['hla'],
                "bp_systolic": int(donor_data['bp_systolic']),
                "bp_diastolic": int(donor_data['bp_diastolic']),
                "creatinine": float(donor_data['creatinine']),
                "bilirubin": float(donor_data['bilirubin']),
                "ejection_fraction": int(donor_data['ejection_fraction']),
                "smoking_history": donor_data['smoking_history'],
                "diabetes": bool(donor_data['diabetes']),
                "bmi": float(donor_data['bmi']),
                "map": float(donor_data['map']),
                "cholesterol": int(donor_data['cholesterol']),
                "glucose_level": int(donor_data['glucose_level']),
                "organ": donor_data['organ'],
                "ocular_pressure": float(donor_data['ocular_pressure']),
                "compatibility_score": float(proba)
            })
        
        # Sort and return top 10
        return sorted(predictions, key=lambda x: x['compatibility_score'], reverse=True)[:10]
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
