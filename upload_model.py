from huggingface_hub import HfApi, login
import os
from pathlib import Path
import shutil
from transformers import AutoTokenizer

# Login to Hugging Face (you'll need to run this interactively first time)
login()

# Initialize the Hugging Face API
api = HfApi()

def prepare_model_files():
    # Create a temporary directory for all files
    temp_dir = Path("./temp_upload")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_path = Path("./phi2-finetuned-final")
    for file in model_path.glob("*"):
        # Skip .git directory and hidden files
        if not file.name.startswith('.') and file.name != '.git':
            try:
                if file.is_file():
                    shutil.copy2(file, temp_dir)
                    print(f"Copied {file.name}")
            except Exception as e:
                print(f"Error copying {file}: {e}")
    
    # Save tokenizer files
    try:
        print("Saving tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        print("Tokenizer files saved successfully")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
    
    return temp_dir

def upload_model_to_hub():
    repo_id = "debasisha/phi2-finetuned"
    
    # Prepare files
    temp_dir = prepare_model_files()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True)
        print(f"Repository {repo_id} created/verified successfully")
    except Exception as e:
        print(f"Repository creation error: {e}")
    
    # Files to upload
    extensions = [
        '.bin', '.json', '.safetensors', 
        'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'config.json'
    ]
    
    # Upload each file
    for ext in extensions:
        for file_path in temp_dir.glob(f'*{ext}'):
            try:
                print(f"Uploading {file_path}...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Successfully uploaded {file_path}")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        print("Starting model upload process...")
        upload_model_to_hub()
        print("Upload process completed successfully")
    except Exception as e:
        print(f"Upload process failed: {e}") 