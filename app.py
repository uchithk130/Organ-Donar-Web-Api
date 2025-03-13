from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict

app = FastAPI()

# Load artifacts and data
donors = pd.read_csv('donors_raw.csv')
model = joblib.load('transplant_rf_model.joblib')
scaler = joblib.load('transplant_scaler.joblib')
blood_encoder = joblib.load('blood_group_encoder.joblib')
organ_encoder = joblib.load('organ_encoder.joblib')

blood_compatibility = {
    'O-': ['O-'],
    'O+': ['O-', 'O+'],
    'A-': ['A-', 'O-'],
    'A+': ['A-', 'A+', 'O-', 'O+'],
    'B-': ['B-', 'O-'],
    'B+': ['B-', 'B+', 'O-', 'O+'],
    'AB-': ['A-', 'B-', 'AB-', 'O-'],
    'AB+': 'ALL'  # AB+ can receive from any blood type
}

class RecipientRequest(BaseModel):
    age: float
    weight: float
    height: float
    blood_group: str
    organ_needed: str
    urgency: int
    bp_systolic: float
    bp_diastolic: float
    creatinine: float
    bilirubin: float
    ejection_fraction: float
    cholesterol: float
    glucose_level: float
    ocular_pressure: float

def preprocess_input(donor: dict, recipient: dict) -> np.array:
    try:
        # Create feature vector in the same order as training
        features = [
            donor['age'], donor['weight'], donor['height'],
            donor['weight'] / (donor['height'] ** 2),
            blood_encoder.transform([[donor['blood_group']]])[0][0],
            organ_encoder.transform([[donor['organ']]])[0][0],
            donor['creatinine'], donor['bilirubin'],
            donor['ejection_fraction'],
            (donor['bp_systolic'] + 2 * donor['bp_diastolic']) / 3,
            donor['cholesterol'], donor['glucose_level'],
            donor['ocular_pressure'],
            # Recipient features
            recipient['age'], recipient['weight'], recipient['height'],
            recipient['weight'] / (recipient['height'] ** 2),
            blood_encoder.transform([[recipient['blood_group']]])[0][0],
            organ_encoder.transform([[recipient['organ_needed']]])[0][0],
            recipient['creatinine'], recipient['bilirubin'],
            recipient['ejection_fraction'],
            (recipient['bp_systolic'] + 2 * recipient['bp_diastolic']) / 3,
            recipient['urgency'],
            recipient['cholesterol'], recipient['glucose_level'],
            recipient['ocular_pressure']
        ]
        return scaler.transform([features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")

@app.post("/match-donors", response_model=List[Dict])
async def match_donors(recipient: RecipientRequest):
    try:
        recipient_data = recipient.dict()

        # Calculate additional fields
        recipient_data['bmi'] = recipient_data['weight'] / (recipient_data['height'] ** 2)
        recipient_data['map'] = (recipient_data['bp_systolic'] + 2 * recipient_data['bp_diastolic']) / 3

        organ = recipient_data['organ_needed']
        blood_group = recipient_data['blood_group']

        # Blood compatibility check
        if blood_compatibility[blood_group] == 'ALL':
            compatible_donors = donors[donors['organ'] == organ]
        else:
            compatible_donors = donors[
                (donors['organ'] == organ) &
                (donors['blood_group'].isin(blood_compatibility.get(blood_group, [])))
            ]

        if compatible_donors.empty:
            return []

        # Prepare data for prediction
        predictions = []
        for _, donor in compatible_donors.iterrows():
            X = preprocess_input(donor.to_dict(), recipient_data)
            proba = model.predict_proba(X)[0][1]
            predictions.append({
                "donor_id": int(donor['donor_id']),  # Assuming each donor has a unique ID column
                "name": donor['name'],
                "age": int(donor['age']),
                "blood_group": donor['blood_group'],
                "compatibility_score": float(proba)
            })

        # Return top 10 donors sorted by compatibility score
        return sorted(predictions, key=lambda x: x['compatibility_score'], reverse=True)[:10]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "healthy"}
