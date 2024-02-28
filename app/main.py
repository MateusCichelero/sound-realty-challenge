from fastapi import FastAPI, Body, HTTPException
from app.models import InputAll, InputMinimal, InferenceOutput
from contextlib import asynccontextmanager
from pydantic import ValidationError
from pandas import read_csv, DataFrame, read_json
from sklearn.pipeline import Pipeline
import pickle
import json


# Load Model and Artifacts with Lifespan Events
def model_predictor(df_input: DataFrame):
    """
    Loads pre-trained model and feature list, processes input data,
    performs prediction, and returns the inference output.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing features for prediction.

    Returns:
        InferenceOutput: Object containing the predicted value.
    """
    with open("app/model/model.pkl", 'rb') as model_file:
        model = pickle.load(model_file) 
    with open('app/model/model_features.json') as json_file:
        features = json.load(json_file)
    demographics = read_csv("app/data/zipcode_demographics.csv",
                      dtype={'zipcode': str})

    
    full_features = df_input.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")

    
    prediction = model.predict(full_features[features])
    return InferenceOutput(prediction=prediction[0])
    

# Life Span Management
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model/ pipeline
    ml_models["model_predictor"] = model_predictor
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(
    title="Housing Prices ML Regression API",
    description="API to perform regression predictions using a Machine Learning model, with two endpoints for different feature sets.",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """
    Root endpoint of the API.
    
    Returns a JSON response with a welcome message for the Housing Prices ML Regression API.
    
    Returns:
        dict: A dictionary containing the key "message" with the welcome message value.
    """
    return {"message": "Welcome to the Housing Prices ML Regression API"}

# Health check endpoint
@app.get("/health")
def check_health():
    """
    Performs a basic health check of the API and returns a response indicating its health status.
    
    This endpoint is intended to be used by external monitoring tools or clients to verify
    if the API is running and operational.
    
    Returns:
        dict: A dictionary containing the following key-value pair:
            - "status": (str) The health status of the API, currently always set to "ok".
    """
    return {"status": "ok"}

@app.post("/predict", response_model=InferenceOutput)
async def predict(data: InputAll = Body(...)):
    """
    Predicts housing prices using a machine learning model.
    
    This endpoint accepts JSON data containing various housing features and returns the predicted
    price using the loaded model.
    
    Args:
        data (InputAll): A Pydantic model containing the complete set of sales features;
    
    Returns:
        InferenceOutput: A Pydantic model containing the predicted price:
            - **prediction:** (float) The predicted housing price
    
    Raises:
        HTTPException: Raises an HTTP exception with the following status codes:
            - 400 (Bad Request): If the input data is invalid (e.g., missing required fields)
            - 500 (Internal Server Error): If any other unexpected error occurs during prediction
    """
    try:
        df_input = DataFrame([data.dict()])
        return ml_models["model_predictor"](df_input)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error handling the request")

@app.post("/predict_minimal", response_model=InferenceOutput)
async def predict_minimal(data: InputMinimal = Body(...)):
    """
    Performs housing price prediction using the loaded ML model based on minimal features provided in the request body.
    
    This endpoint expects a request body containing an `InputMinimal` pydantic model instance and returns a
    response with the predicted housing price as a JSON object.
    
    Args:
        data: (InputMinimal) The minimal features for the housing price prediction.
    
    Returns:
        InferenceOutput: A JSON response containing the predicted housing price.
    
    Raises:
        HTTPException: Raises an HTTP exception with the following status codes:
            - 400 (Bad Request): If the input data is invalid (e.g., missing required fields)
            - 500 (Internal Server Error): If any other unexpected error occurs during prediction
    """
    try:
        df_input = DataFrame([data.dict()])
        return ml_models["model_predictor"](df_input)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error handling the request")
