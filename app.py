from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("Error loading model:", e)


# Home route
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction route
@app.post("/predict_form")
def predict_form(
    request: Request,
    pclass: int = Form(...),
    sex: int = Form(...),
    age: float = Form(...),
    fare: float = Form(...)
):
    try:
        if model is None:
            raise Exception("Model not loaded properly")

        # Prepare input data
        data = np.array([[pclass, sex, age, fare]])

        # Prediction
        prediction = model.predict(data)[0]

        # Optional probability (if model supports it)
        try:
            prob = model.predict_proba(data)[0][1]
            probability_text = f" (Confidence: {prob:.2f})"
        except:
            probability_text = ""

        # Final result
        if prediction == 1:
            result = f"Survived 🎉{probability_text}"
        else:
            result = f"Not Survived ❌{probability_text}"

    except Exception as e:
        result = f"Error: {str(e)}"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result
    })


# Optional: health check route (useful for deployment)
@app.get("/health")
def health():
    return {"status": "OK"}
