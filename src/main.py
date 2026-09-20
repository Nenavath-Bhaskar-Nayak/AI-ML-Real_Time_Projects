from fastapi import FastAPI
from src.schemas import StudentData
from src.predict import StudentPredictor
app = FastAPI(
    title="Student Performance Prediction API",
    description="ML API for Student Pass/Fail Prediction",
    version="1.0.0"
)
# Load the model once at startup
predictor = StudentPredictor()
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(student: StudentData):
    result = predictor.predict_single(student.model_dump())
    return result

# Run with: uvicorn src.main:app --reload