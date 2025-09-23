from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
# from google.colab import userdata

# Access the token from Colab secrets
# HF_TOKEN = userdata.get('HF_TOKEN')
# api = HfApi(token=HF_TOKEN)

# Access the token from Colab secrets and set it as an environment variable
# os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

HF_TOKEN = os.getenv('HF_TOKEN')
api = HfApi(token=HF_TOKEN)


repo_id = "ranjithkumarsundaramoorthy/tourism-project"    # please create your space and repository

repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data/",
    repo_id=repo_id,
    repo_type=repo_type,
)
