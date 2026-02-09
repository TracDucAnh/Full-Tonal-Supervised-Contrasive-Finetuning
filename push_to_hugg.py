# This code is responsible for pushing the checkpoints of finetuned Wav2vec and HuBERT to HuggingFace model hub.
import os
from dotenv import load_dotenv

# Configure writable Hugging Face cache paths before importing huggingface_hub.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_home")

os.environ.setdefault("HF_HOME", LOCAL_HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(os.environ["HF_HOME"], "assets"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.environ["HF_HOME"], "xdg"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_ASSETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

from huggingface_hub import HfApi, create_repo

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Retrieve the token (ensure the key in .env is HF_TOKEN)
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("[ERROR] HF_TOKEN not found in .env file. Please check your configuration.")
        return

    # Initialize the API
    api = HfApi(token=hf_token)
    print(f"[INFO] HF_HOME: {os.environ['HF_HOME']}")
    print(f"[INFO] HF_HUB_DISABLE_XET: {os.environ.get('HF_HUB_DISABLE_XET')}")

    try:
        # Get user info to determine the namespace (username)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[INFO] Successfully logged in as: {username}")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return

    # Configuration for the models to be pushed
    # Paths are based on the provided screenshot structure
    models_to_push = [
        {
            "local_path": "Wav2vec_finetuned/checkpoints/best_model",
            "repo_name": "wav2vec_tone"
        },
        {
            "local_path": "HuBERT_finetuned/checkpoints/best_model",
            "repo_name": "hubert_tone"
        }
    ]

    for model in models_to_push:
        repo_id = f"{username}/{model['repo_name']}"
        local_dir = model['local_path']

        print("--------------------------------------------------")
        print(f"[INFO] Processing repo: {repo_id}")

        # Check if local directory exists
        if not os.path.exists(local_dir):
            print(f"[WARNING] Local directory '{local_dir}' does not exist. Skipping.")
            continue

        try:
            # Create repo on Hugging Face if it doesn't exist
            # private=True creates a private repo (change to False for public)
            create_repo(repo_id=repo_id, token=hf_token, private=False, exist_ok=True)
            print(f"[INFO] Repo created or already exists: {repo_id}")

            # Upload the directory content
            print(f"[INFO] Uploading data from '{local_dir}'...")
            
            api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload best model checkpoint"
            )
            
            print(f"[SUCCESS] Upload completed for: {repo_id}")
            
        except Exception as e:
            print(f"[ERROR] An error occurred while processing {repo_id}: {e}")

    print("--------------------------------------------------")
    print("[INFO] Script finished.")

if __name__ == "__main__":
    main()
