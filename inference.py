from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")


@app.get("/")
def root():
    return {
        "message": "Iris ML inference service is running"
    }


@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]])

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }