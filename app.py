from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI()

# ---------- Input schema ----------
class HouseInput(BaseModel):
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    ocean_proximity: str


# ---------- Lazy model loading (CRITICAL for free tier) ----------
model = None

def get_model():
    global model
    if model is None:
        with open("house_model.pkl", "rb") as f:
            model = pickle.load(f)
    return model


# ---------- Routes ----------
@app.get("/")
def home():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: HouseInput):
    model = get_model()

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Log transform numerical columns
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        df[col] = np.log(df[col] + 1)

    # Feature engineering
    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["household_ratio"] = df["total_rooms"] / df["households"]

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    return {
        "predicted_price": float(prediction[0])
    }


# ---------- Local run (Render ignores this) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=1
    )
