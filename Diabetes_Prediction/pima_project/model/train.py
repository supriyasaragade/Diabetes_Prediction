# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")

# Define constants for the dataset and output paths
api = HfApi(token=HF_TOKEN)

Xtrain_path = "hf://datasets/supriyasaragade/Diabetes-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/supriyasaragade/Diabetes-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/supriyasaragade/Diabetes-Prediction/ytrain.csv"
ytest_path = "hf://datasets/supriyasaragade/Diabetes-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# define feature lists (reuse variables already in the notebook)
skewed_numeric_features = ['Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
symmetric_numeric_features = ['Glucose', 'BloodPressure'] 
categorical_features = ['Nutritional_Status']

# transformers
skewed_numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
symmetric_numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# column transformer and full pipeline
preprocessor = ColumnTransformer([
    ('skewed_num', skewed_numeric_transformer, skewed_numeric_features),
    ('symmetric_num', symmetric_numeric_transformer, symmetric_numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

clf = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 3, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': [None, 'balanced']
}
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_diabetes_prediction.joblib")

# Upload to Hugging Face
repo_id = "supriyasaragade/Diabetes-Prediction"
repo_type = "model"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_diabetes_prediction", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_diabetes_prediction.joblib",
    path_in_repo="best_diabetes_prediction.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
