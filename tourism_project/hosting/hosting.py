from huggingface_hub import HfApi
import os

# os.environ["HF_TOKEN"] = "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"    # please use your token
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="/content/tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="ranjithkumarsundaramoorthy/tourism-project-UI",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
