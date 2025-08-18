import os
import argparse
import json
from tabulate import tabulate
import torchtext
from torchtext.data.utils import get_tokenizer

# Import all necessary modules
# Note: The original code implies these modules exist in your project structure.
from rnn_preprocessing.embeddings import get_glove_embeddings
from rnn_preprocessing.encoder import SentenceEncoder
from fix.fix import LLM_Fixer
from checks.redundancy_checks import check_redundancy
from checks.pii_checks import check_pii_and_secrets
from checks.completeness_check import check_completeness
from checks.contradiction_checks import check_conflicting_instructions


def get_validation_checks():
    """
    Returns a list of all validation check functions.
    """
    return [
        check_redundancy,
        check_pii_and_secrets,
        check_completeness,
        check_conflicting_instructions
    ]

def run_validation(directory_path):
    """
    Runs all validation checks on prompts in a given directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    report = []
    validation_checks = get_validation_checks()
    
    # Initialize GloVe once
    try:
        glove = get_glove_embeddings()
    except Exception as e:
        print(f"Failed to load GloVe embeddings: {e}")
        return [] # Exit if embeddings fail to load

    # Corrected instantiation of SentenceEncoder
    # Access the vocabulary size and vectors from the loaded glove object
    vocab_size = len(glove.stoi)
    glove_vectors = glove.vectors
    tokenizer = get_tokenizer('basic_english')
    encoder = SentenceEncoder(embedding_dim=100, hidden_dim=64, vocab_size=vocab_size, glove_vectors=glove_vectors)

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            prompt_content = "" # Initialize prompt_content for each file

            # Try common encodings in order
            encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']
            
            for encoding in encodings_to_try:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        prompt_content = f.read()
                    print(f"Successfully read {filename} with encoding: {encoding}")
                    break # Successfully read, exit loop
                except UnicodeDecodeError:
                    print(f"Failed to read {filename} with encoding: {encoding}")
                    continue # Try next encoding
                except Exception as e: # Catch other potential file errors
                    print(f"An unexpected error occurred while reading {filename} with encoding {encoding}: {e}")
                    prompt_content = "" # Clear content if error
                    break # Stop trying encodings for this file
            
            if not prompt_content:
                print(f"Warning: Could not decode {filename} with any tried encoding. Skipping file.")
                continue # Skip to the next file if decoding failed

            issues_found = []
            for check_func in validation_checks:
                if check_func.__name__ == 'check_redundancy':
                    # Ensure GloVe is loaded before calling redundancy check
                    if glove:
                        issues_found.extend(check_func(prompt_content, encoder, glove.stoi, tokenizer))
                else:
                    issues_found.extend(check_func(prompt_content))
            
            # Always add to report, even if no issues, so it appears in the table
            report.append({
                "file": filename,
                "issues": issues_found,
                "original_content": prompt_content # Store original content for fixing later
            })
    return report

def generate_report(report_data, report_format, output_file=None):
    """
    Generates a report in JSON or CLI table format, and optionally saves it to a file.
    """
    # Prepare data for JSON output (remove original_content)
    clean_report_data = []
    for entry in report_data:
        new_entry = {k: v for k, v in entry.items() if k != 'original_content'}
        clean_report_data.append(new_entry)

    if report_format == "json" or report_format == "all":
        json_output = json.dumps(clean_report_data, indent=4)
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"\nReport saved to: {output_file}")
            except Exception as e:
                print(f"Error saving JSON report to {output_file}: {e}")
        else:
            print("\n--- JSON Report ---")
            print(json_output)
            print("--- End JSON Report ---")

    if report_format == "cli" or report_format == "all":
        table_data = []
        for entry in report_data:
            if not entry['issues']:
                table_data.append([entry['file'], "No issues", "N/A", "N/A"])
            else:
                for issue in entry["issues"]:
                    table_data.append([
                        entry["file"],
                        issue["issue_type"],
                        issue["description"],
                        issue.get("suggested_fix", "N/A")
                    ])
        
        headers = ["File Name", "Issue Type", "Description", "Suggested Fix"]
        cli_output = tabulate(table_data, headers=headers, tablefmt="grid")
        
        print("\n--- CLI Report ---")
        print(cli_output)
        print("--- End CLI Report ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Directory Validator Module")
    parser.add_argument("directory", help="Path to the directory containing .txt prompts.")
    parser.add_argument("--report", action="store_true", help="Generate a report of issues.")
    parser.add_argument("--fix", action="store_true", help="Auto-fix prompts with a local LLM.")
    parser.add_argument("--output-format", 
                        choices=["cli", "json", "all"], 
                        default="cli", 
                        help="Specify report output format (cli, json, or all).")
    parser.add_argument("--output-file", 
                        help="Optional: Path to save the JSON report (e.g., report.json). Only applicable for 'json' or 'all' output formats.")
    args = parser.parse_args()

    # Run validation first to get all issues
    validation_report = run_validation(args.directory)

    # Conditionally instantiate the LLM_Fixer and apply fixes
    if args.fix:
        try:
            llm_fixer = LLM_Fixer()
        except Exception as e:
            print(f"Could not initialize LLM Fixer: {e}")
            llm_fixer = None

        if llm_fixer:
            corrected_prompts_dir = os.path.join(args.directory, "corrected_prompts")
            os.makedirs(corrected_prompts_dir, exist_ok=True)
            print(f"\nCorrected prompts will be saved in: {corrected_prompts_dir}")

            for entry in validation_report:
                filename = entry["file"]
                original_prompt = entry["original_content"] # Retrieve original content from report entry

                if entry['issues']:
                    print(f"Attempting to fix: {filename}")
                    fixed_prompt = llm_fixer.fix_prompt_with_llm(original_prompt, entry["issues"])
                    
                    # Update the report with the suggested fix for display
                    for issue in entry['issues']:
                        issue['suggested_fix'] = fixed_prompt
                    
                    # Save the fixed content to the new directory
                    corrected_filepath = os.path.join(corrected_prompts_dir, filename)
                    try:
                        with open(corrected_filepath, 'w', encoding='utf-8') as f:
                            f.write(fixed_prompt)
                        print(f"Fixed and saved to: {corrected_filepath}")
                    except Exception as e:
                        print(f"Error writing fixed prompt to {corrected_filepath}: {e}")
                else:
                    # If no issues, copy the original prompt to the corrected_prompts folder
                    # This ensures all processed prompts have a corresponding file in the output folder
                    corrected_filepath = os.path.join(corrected_prompts_dir, filename)
                    try:
                        with open(corrected_filepath, 'w', encoding='utf-8') as f:
                            f.write(original_prompt)
                        print(f"No issues found for {filename}. Copied to: {corrected_filepath}")
                    except Exception as e:
                        print(f"Error copying original prompt to {corrected_filepath}: {e}")
    
    # Generate the report at the end
    if args.report:
        generate_report(validation_report, args.output_format, args.output_file)