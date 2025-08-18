# In checks/pii_checks.py
import re
from typing import List
def check_pii_and_secrets(prompt_content: str) -> List[dict]: # Changed return type
    issues = []
    
    # Example PII/Secret patterns (add more as needed)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    password_keywords = ["password", "secret_key", "api_key", "auth_token"]
    pii_keywords = ["phone number", "social security number", "ssn", "credit card", "bank account"] # New
    
    # ...
    
    for keyword in pii_keywords: # Check for PII keywords
        if keyword in prompt_content.lower():
            issues.append({
                "issue_type": "PII/Secret",
                "description": f"Keyword '{keyword}' detected, indicating a request for or presence of PII.",
                "suggested_fix": "Avoid including or requesting sensitive personal identifiable information (PII) in prompts."
            })
    if re.search(email_pattern, prompt_content):
        issues.append({
            "issue_type": "PII/Secret",
            "description": "Potential email address detected.",
            "suggested_fix": "Remove or redact personal identifiable information (PII)."
        })
    
    if re.search(phone_pattern, prompt_content):
        issues.append({
            "issue_type": "PII/Secret",
            "description": "Potential phone number detected.",
            "suggested_fix": "Remove or redact personal identifiable information (PII)."
        })

    for keyword in password_keywords:
        if keyword in prompt_content.lower():
            issues.append({
                "issue_type": "PII/Secret",
                "description": f"Keyword '{keyword}' detected, indicating a potential secret.",
                "suggested_fix": "Ensure no sensitive credentials or secrets are hardcoded in the prompt."
            })
    
    return issues # Always return a list, even if empty