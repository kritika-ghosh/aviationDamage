from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("app/mod1.pkl")

# Define input structure
class InputData(BaseModel):
    Investigation_Type: str
    Schedule: str
    Total_Uninjured: int
    Injury_Severity: str
    Engine_Type: str

@app.post("/predict")
def predict(data: InputData):
    # Manually recreate the features
    features = []

    # Investigation.Type_Incident
    features.append(1 if data.Investigation_Type == "Incident" else 0)

    # Schedule_SCHD
    features.append(1 if data.Schedule == "SCHD" else 0)

    # Total.Uninjured
    features.append(data.Total_Uninjured)

    # Injury.Severity_freq — mock logic for now
    # NOTE: Ideally, you save a dict mapping original values to frequencies used in training
    severity_map = {
        "Fatal": 500,  # Replace with actual counts from your df
        "Serious": 400,
        "Minor": 300,
        "Non": 200
    }
    features.append(severity_map.get(data.Injury_Severity, 0))

    # Engine.Type_Turbo Fan
    features.append(1 if data.Engine_Type == "Turbo Fan" else 0)

    # Make prediction
    pred = model.predict([features])[0]

    return {"predicted_damage_class": int(pred)}
