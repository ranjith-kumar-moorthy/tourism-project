from huggingface_hub import HfApi
import os
# from google.colab import userdata

# Access the token from Colab secrets ( key symbol )
# HF_TOKEN = userdata.get('HF_TOKEN')
# api = HfApi(token=HF_TOKEN)

# Access the token from Colab secrets and set it as an environment variable
# os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

HF_TOKEN = os.getenv('HF_TOKEN')
api = HfApi(token=HF_TOKEN)


api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="ranjithkumarsundaramoorthy/tourism-project-UI",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
