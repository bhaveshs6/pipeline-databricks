# Databricks notebook source
import requests
import json

# Define the URL of the Databricks serving endpoint
endpoint_url = "https://adb-6724577987585661.1.azuredatabricks.net/serving-endpoints/titanic/invocations"

# Define the input data as a Python dictionary
input_data = {
    "dataframe_records": [
        {
            "Pclass": 1,
            "Age": 35,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 53.1
        },
        {
            "Pclass": 3,
            "Age": 22,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.25
        }
    ]
}

# Convert the input data to a JSON string
json_data = json.dumps(input_data)

# Define your Databricks token
databricks_token = dbutils.secrets.get(scope="tokens", key="dbtoken")

# Set up the HTTP headers
headers = {
    "Content-Type": "application/json"
}

# Send a POST request to the endpoint with the input data and authentication
response = requests.post(
    endpoint_url,
    data=json_data,
    headers=headers,
    auth=("token", databricks_token)
)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response to get predictions
    response_data = json.loads(response.text)
    predictions = response_data.get("predictions", [])
    print("Predictions:", predictions)
else:
    print("Request failed with status code:", response.status_code)
    print("Response content:", response.text)


# COMMAND ----------


