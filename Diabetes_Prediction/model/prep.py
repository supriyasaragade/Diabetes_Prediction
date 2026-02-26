# for data manipulation
import pandas as pd
import numpy as np
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=HF_TOKEN)

# Load dataset from the correct path
DATASET_PATH = "hf://datasets/supriyasaragade/Diabetes-Prediction/diabetes.csv"
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

def calculate_nutritional_status(x): 
    if x == 0.0: 
        return np.nan
    elif x < 18.5: 
        return "Underweight"
    elif x < 25: 
        return "Normal"
    elif x >= 25 and x < 30: 
        return "Overweight"
    elif x >= 30: 
        return "Obese"

data['Nutritional_Status'] = data['BMI'].apply(calculate_nutritional_status)
data.drop('BMI', axis=1, inplace=True)

# Identify columns with zero values that should be treated as missing
columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin'] 

# Replace zero values with NaN in the identified columns
data[columns_with_zero_as_missing] = data[columns_with_zero_as_missing].replace(0, np.nan)

target_col = 'Outcome'

# Split into X (features) and y (target)
X = data.drop(columns=[target_col])
y = data[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sandeep-raghuwanshi28/Diabetes-Prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path}")
