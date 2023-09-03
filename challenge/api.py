import fastapi
import joblib

app = fastapi.FastAPI()

# Load the pre-trained XGBoost model
#model = joblib.load("model.joblib")

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:

    return