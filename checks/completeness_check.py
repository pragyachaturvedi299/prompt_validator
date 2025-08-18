# completeness_check.py
from typing import List, Dict # Import List and Dict for type hinting

def check_completeness(prompt: str) -> List[Dict]:
    """
    Checks for common signs of an incomplete prompt (e.g., missing specific context).
    This is a more subjective, rule-based check.
    """
    issues = []
    prompt_lower = prompt.lower()
    
    # Existing check (won't trigger for Prompt 1)
    if "write a report" in prompt_lower and not ("on " in prompt_lower or "about " in prompt_lower or "regarding " in prompt_lower):
        issues.append({
            "issue_type": "Completeness",
            "description": "Prompt asks to 'write a report' but lacks a clear topic (e.g., 'on X', 'about Y').",
            "suggested_fix": "Specify the subject or topic of the report."
        })
    
    # New: Check for missing structure/sections in a "guide" or "report"
    if ("write a guide" in prompt_lower or "write a report" in prompt_lower) and not any(
        section_keyword in prompt_lower for section_keyword in ["sections", "structure", "outline", "table of contents", "chapters"]
    ):
        issues.append({
            "issue_type": "Completeness",
            "description": "Prompt asks for a guide/report but doesn't specify desired sections or structure.",
            "suggested_fix": "Define the required sections, outline, or structure for the guide/report."
        })

    # New: Check for missing audience/tone
    if ("write" in prompt_lower or "explain" in prompt_lower) and not any(
        audience_keyword in prompt_lower for audience_keyword in ["for a", "to a", "audience", "tone", "style"]
    ):
        issues.append({
            "issue_type": "Completeness",
            "description": "Prompt asks to write/explain something but doesn't specify the target audience or desired tone/style.",
            "suggested_fix": "Specify the target audience (e.g., 'for beginners', 'for experts') and desired tone (e.g., 'formal', 'casual')."
        })
    # Check 2: Missing requirements/scenarios for test plan
    if "create a test plan" in prompt_lower and not ("requirements" in prompt_lower or "scenarios" in prompt_lower or "features" in prompt_lower):
        issues.append({
            "issue_type": "Completeness",
            "description": "Prompt asks to 'create a test plan' but lacks specified requirements, features, or scenarios.",
            "suggested_fix": "Provide the requirements, features, or specific scenarios the test plan should cover."
        })
    
    # Add more completeness checks as needed
    # Example: Check for missing output format
    if ("generate" in prompt_lower or "create" in prompt_lower) and not any(
        fmt in prompt_lower for fmt in ["json", "xml", "csv", "markdown", "table", "bullet points", "paragraph"]
    ):
        issues.append({
            "issue_type": "Completeness",
            "description": "Prompt asks to generate/create content but doesn't specify the desired output format.",
            "suggested_fix": "Specify the desired output format (e.g., 'in JSON format', 'as a table')."
        })

    return issues # Always return a list, even if empty