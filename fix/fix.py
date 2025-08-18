# In prompt_validator/fix/fix.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict

# Define the local folder path relative to the project root
# This path should point to where you saved the Phi-2 model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..", "model")

class LLM_Fixer:
    """
    Manages the loading and usage of a local LLM for prompt fixing.
    The model is loaded only once upon class instantiation.
    """
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the LLM and tokenizer from the local path.
        This method is called only once during initialization.
        """
        try:
            # Check if the model path exists
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

            print(f"Loading LLM from local directory: {MODEL_PATH}...")
            
            # Use quantization to reduce memory footprint
            # Check for bfloat16 support
            compute_dtype = torch.bfloat16
            if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8: # Check for Ampere or newer
                print("Warning: bfloat16 not supported on this GPU or CUDA not available. Falling back to float16 for quantization compute_dtype.")
                compute_dtype = torch.float16 # Fallback to float16 if bfloat16 is not supported

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4", # Recommended for 4-bit quantization
                bnb_4bit_use_double_quant=True, # Recommended for 4-bit quantization
            )

            # Load tokenizer and model, ensuring trust_remote_code=True for Phi-2
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=quantization_config,
                device_map='auto',
                trust_remote_code=True # Crucial for Phi-2
            )
            
            # Set pad_token_id if not already set (common for generation)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("LLM loaded successfully.")

        except Exception as e:
            print(f"Error loading local LLM: {e}")
            self.tokenizer = None
            self.model = None

    def fix_prompt_with_llm(self, prompt: str, issues: List[Dict]) -> str:
        """
        Uses the loaded LLM to generate a fixed version of a prompt.
        """
        if not self.model or not self.tokenizer:
            print("LLM is not available. Returning original prompt.")
            return prompt

        if not issues:
            return prompt  # No fixes needed

        issue_descriptions = "\n".join([
            f"- {issue['issue_type']}: {issue['description']}. Suggested fix: {issue.get('suggested_fix', 'N/A')}"
            for issue in issues
        ])

        # Crafting the prompt for the LLM
        # It's often better to put the instruction first, then the context.
        # Also, explicitly ask for "Corrected Prompt:" to make parsing easier.
        llm_prompt = (
            f"You are an expert prompt engineer. Your task is to fix the following prompt "
            f"based on the identified issues. Provide only the corrected prompt text, nothing else.\n\n"
            f"Issues to fix:\n---\n{issue_descriptions}\n---\n\n"
            f"Original Prompt:\n---\n{prompt}\n---\n\n"
            f"Corrected Prompt:\n" # The model should complete from here
        )

        try:
            # Encode the prompt
            inputs = self.tokenizer(llm_prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)
            
            # Generate the response
            # Use max_new_tokens to control the length of the *generated* part
            # Set pad_token_id for generation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256, # Max tokens for the *generated* response
                do_sample=False,    # For deterministic output
                temperature=0.0,    # For deterministic output
                pad_token_id=self.tokenizer.pad_token_id, # Important for batching and generation
                eos_token_id=self.tokenizer.eos_token_id # Stop generation at EOS token
            )
            
            # Decode the entire output sequence
            full_response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (after the input prompt)
            # This assumes the model will directly continue from "Corrected Prompt:\n"
            # You might need to refine this parsing based on actual model behavior
            if full_response_text.startswith(llm_prompt):
                fixed_prompt = full_response_text[len(llm_prompt):].strip()
            else:
                # Fallback if the model doesn't perfectly echo the input prompt
                # This can happen if the model's generation starts differently
                print("Warning: Model response did not start with the expected prompt prefix. Attempting alternative parsing.")
                # A more robust parsing might involve looking for a specific start/end marker
                # or just taking the last N tokens if the prompt is very long.
                fixed_prompt = full_response_text.strip() # Take the whole thing and hope for the best
            
            # Remove any trailing instruction from the model if it continues to chat
            # This is a heuristic, might need adjustment
            if "Original Prompt:" in fixed_prompt:
                fixed_prompt = fixed_prompt.split("Original Prompt:")[0].strip()
            if "Issues to fix:" in fixed_prompt:
                fixed_prompt = fixed_prompt.split("Issues to fix:")[0].strip()

            return fixed_prompt
        except Exception as e:
            print(f"Error generating fix with LLM: {e}")
            return prompt  # Return original prompt on failure