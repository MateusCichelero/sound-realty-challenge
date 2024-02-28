# Housing Prices ML Regression API

- **phData Machine Learning Engineer Candidate Project**
- Author: Mateus Cichelero
- Date: February 2024

This project provides a REST API that uses a machine learning regression model to predict housing prices for the Sound Realty Company. The API allows users to input housing characteristics and receive a price prediction, supporting Sound Realty in their valuation and decision-making processes.

## Technologies

- Python: The primary programming language.
- FastAPI: Framework for rapidly building web APIs.
- Scikit-Learn: Tools for machine learning, including the regression model.
- Pandas: Library for data manipulation and analysis.
- Poetry: For managing Python dependencies.
- Docker: Containerization for easy development and deployment.

## Features

- Endpoint /predict: Accepts comprehensive housing data and returns a price prediction.
- Endpoint /predict_minimal: Provides predictions using a core set of features.
- Data Validation: Ensures input data aligns with model expectations.
- Swagger Documentation: Automatic API documentation for ease of use.

## Challenge Requirements
- [x] Deploy the model as an endpoint on a RESTful service which receives JSON POST data. (see **/predict endpoint at app/main**)
- [x] Create an additional API endpoint where only the required features have to be provided in order to get a prediction. (see **/predict_minimal endpoint at app/main**) 
- [x] Create a test script which submits examples to the endpoint to demonstrate its behavior. (try **python experiments/api_test_script.py** with the api application running on docker)
- [x] Evaluate the performance of the basic model. (see regression metrics comment at https://github.com/MateusCichelero/sound-realty-challenge/pull/11 - use of cml from iterative (https://cml.dev/). Implemented at experiments/create_model.py in **experiments-basic-model** branch)
- [x] Improve the model by applying some basic machine-learning principles. Check the implementation of catboost regressor with grid search and using categorical features in the **experiments-gbm-model** branch. See regression metrics comment at https://github.com/MateusCichelero/sound-realty-challenge/pull/12 (a good improvement compared to the basic model)

## Basic Usage:

## Create environment with Poetry
With poetry alread installed and at the root of this repository:

```bash
> poetry install
> poetry shell
```

after having the env ready, you can run the API locally with:
```bash
uvicorn app.main:app --reload
```

## Using the API with Docker
With docker engine already installed:

```bash
> docker build -t phdata-api .
> docker run -d --name mycontainer -p 80:80 phdata-api
```

After that you should be able to access the API docs from **http://0.0.0.0/docs** (and to apply the test script from **experiments/api_test_script.py**)