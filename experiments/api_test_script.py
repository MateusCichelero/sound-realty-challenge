import requests
import random
import json
import pandas as pd

# Loads unseen examples csv file
data = pd.read_csv("./data/future_unseen_examples.csv", dtype={'zipcode': str})

# Randomly picks 20 records for the test
sample_cases = data.sample(20)

# URL as set from docker 
url = "http://0.0.0.0:80/predict"

# Headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# Loop para enviar os registros
for case in sample_cases.to_dict(orient="records"):
    # Converter para JSON
    payload = json.dumps(case)

    # Enviar requisição POST
    response = requests.post(url, headers=headers, data=payload)

    # Verificar o status code
    if response.status_code == 200:
        print(f"Success! Case: {case}")
        print(f"Response: {response.text}")
        print("###############################################")
    else:
        print(f"Error! Case: {case}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        print("###############################################")