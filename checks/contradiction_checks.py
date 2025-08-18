# contradiction_checks.py
import re
from typing import List, Dict

def check_conflicting_instructions(prompt: str) -> List[Dict]:
    """Checks for contradictory instructions within a prompt."""
    issues = []
    prompt_lower = prompt.lower()
    
    conflicting_pairs = [
        ("no longer than", "at least"),
        ("brief", "comprehensive"),
        ("short", "long"),
        ("include", "exclude"),
        ("only", "and also"),
        ("must not", "must"),
        ("no longer than", "comprehensive"), 
        ("no longer than", "detailed"),      
        ("no longer than", "extensive"),     
        ("no longer than", "in-depth"),   
    ]
    
    for pair in conflicting_pairs:
        phrase1, phrase2 = pair
        if phrase1 in prompt_lower and phrase2 in prompt_lower:
            issues.append({
                "issue_type": "Contradiction",
                "description": f"Conflicting instructions detected: '{phrase1}' and '{phrase2}' are both present.",
                "suggested_fix": "Clarify the instructions to remove contradictory requirements."
            })
            
    # Simple check for number-based conflicts
    numbers = [int(n) for n in re.findall(r'\d+', prompt)]
    
    # Filter out zeros from the numbers list if you don't want them to affect min()
    # Or, more simply, ensure min(numbers) is not zero before division.
    
    if len(numbers) >= 2:
        # Ensure min(numbers) is not zero BEFORE attempting division
        # This is the crucial change: reorder the conditions
        if min(numbers) > 0 and max(numbers) / min(numbers) > 100:
             issues.append({
                "issue_type": "Contradiction",
                "description": f"Large numerical discrepancy detected ({min(numbers)} vs {max(numbers)}), potentially indicating conflicting quantity instructions.",
                "suggested_fix": "Review numerical constraints to ensure they are consistent and not contradictory."
            })
        
    return issues