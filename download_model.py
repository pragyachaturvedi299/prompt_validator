import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define the model you want to download
model_id = "microsoft/Phi-2"

# 2. Define the local folder path
# Create the 'model' folder in your project root
model_dir = os.path.join(os.getcwd(), "model")
os.makedirs(model_dir, exist_ok=True)

print(f"Starting download of model '{model_id}' to '{model_dir}'...")

# 3. Load and save the model
# The .from_pretrained() method automatically downloads the model
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 4. Save the model and tokenizer to the specified directory
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("\nDownload complete! The model and tokenizer are saved in the 'model' folder.")