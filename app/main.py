from fastapi import FastAPI, Body, HTTPException
from models import InputAll, InputMinimal, InferenceOutput
from contextlib import asynccontextmanager
from pydantic import ValidationError
from pandas import read_csv, DataFrame, read_json
from sklearn.pipeline import Pipeline
import pickle
import json


# Load Model and Artifacts with Lifespan Events
def model_predictor(df_input: DataFrame):
    with open("./model/model.pkl", 'rb') as model_file:
        model = pickle.load(model_file) 
    with open('./model/model_features.json') as json_file:
        features = json.load(json_file)
    demographics = read_csv("./data/zipcode_demographics.csv",
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


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Housing Prices ML Regression API"}

# Health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}

@app.post("/predict", response_model=InferenceOutput)
async def predict(data: InputAll = Body(...)):
    try:
        df_input = DataFrame([data.dict()])
        return ml_models["model_predictor"](df_input)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro ao processar a requisição")

@app.post("/predict_minimal", response_model=InferenceOutput)
async def predict_minimal(data: InputMinimal = Body(...)):
    #try:
    df_input = DataFrame([data.dict()])
    return ml_models["model_predictor"](df_input)
    #except ValidationError as e:
    #    raise HTTPException(status_code=400, detail=str(e))
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail="Erro ao processar a requisição")
