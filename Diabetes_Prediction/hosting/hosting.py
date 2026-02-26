from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
# Define constants for the dataset and output paths
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="pima_project/deployment",     # the local folder containing your files
    repo_id="supriyasaragade/Diabetes-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
