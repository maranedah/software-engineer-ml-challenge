import fastapi
import joblib
import pandas as pd
from .preprocess import get_top_10_features

app = fastapi.FastAPI()

# Load the pre-trained XGBoost model
#model = joblib.load("model.joblib")

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

def transform_column_to_dummy(df):
    columns = [f"{k}_{v}" for k,v in df["flights"][0].items()]
    return columns

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    columns = transform_column_to_dummy(data)
    # initialize pd frame
    df = pd.DataFrame(0, index=range(1), columns=model._schema)
    for column in columns:
        if column in df.columns:
            df[column] = 1
        else:
            raise fastapi.HTTPException(status_code=400, detail="Attempted to assign column not present in data schema")
    df = get_top_10_features(df)
    prediction = model.predict(df)
    response = {"predict": prediction}
    return response