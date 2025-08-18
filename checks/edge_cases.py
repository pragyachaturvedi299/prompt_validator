# edge_cases.py
from typing import List, Dict # Import List and Dict for type hinting

def check_missing_edge_cases(prompt: str) -> List[Dict]:
    """
    Checks if a test plan-related prompt is missing common edge case considerations.
    """
    issues = []
    prompt_lower = prompt.lower()
    
    test_keywords = {"test plan", "test case", "requirements", "criteria", "testing", "qa"}
    edge_case_keywords = {"edge case", "invalid", "negative", "boundary", "null", "empty", "error handling", "unusual input", "extreme values"}
    
    # Check if it's a test-related prompt AND it doesn't mention edge cases
    is_test_related = any(word in prompt_lower for word in test_keywords)
    mentions_edge_cases = any(word in prompt_lower for word in edge_case_keywords)

    if is_test_related and not mentions_edge_cases:
        issues.append({
            "issue_type": "Completeness (Edge Cases)",
            "description": "The prompt is related to testing/test plans but does not explicitly mention considering edge cases, invalid inputs, or boundary conditions.",
            "suggested_fix": "Explicitly instruct the model to consider edge cases, invalid inputs, boundary conditions, and error handling scenarios in the test plan/cases."
        })
        
    return issues # Always return a list, even if empty