from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import numpy as np
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_form")
def predict_form(request: Request,
                 Pclass: int = Form(...),
                 Sex: int = Form(...),
                 Age: float = Form(...),
                 Fare: float = Form(...)):

    data = np.array([[Pclass, Sex, Age, Fare]])
    prediction = model.predict(data)[0]

    result = "Survived 🎉" if prediction == 1 else "Not Survived ❌"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result
    })