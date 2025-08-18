# In checks/redundancy_checks.py

import re
import torch
from collections import Counter
from typing import List
# from rnn_preprocessing.encoder import SentenceEncoder # You might need this import if SentenceEncoder is not globally accessible

# This is the new, more robust check_redundancy function
def check_redundancy(prompt_content: str, encoder, glove_stoi: dict, tokenizer) -> List[dict]:
    """
    Checks for redundancy in the prompt content using sentence embeddings.
    Identifies sentences that are semantically very similar.

    Args:
        prompt_content (str): The content of the prompt.
        encoder (SentenceEncoder): The trained SentenceEncoder model.
        glove_stoi (dict): The string-to-index vocabulary from GloVe.
        tokenizer (callable): The tokenizer function (e.g., basic_english).

    Returns:
        list: A list of dictionaries, each describing a redundancy issue.
    """
    issues = []
    
    # Basic sentence splitting - consider using NLTK for more robust splitting
    # if you have it installed:
    # import nltk
    # nltk.download('punkt') # run once
    # sentences = nltk.sent_tokenize(prompt_content)
    
    # For now, using a simple split by common sentence terminators
    sentences = re.split(r'(?<=[.!?\n])\s*', prompt_content.strip()) # Added \n to the split pattern
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return issues # Not enough sentences to check for redundancy

    # Prepare data for the encoder
    tokenized_sentences = [tokenizer(s) for s in sentences]
    
    # Convert tokens to numerical IDs using glove_stoi
    # Handle unknown words by mapping them to the <unk> token ID, or 0 if <unk> isn't found
    indexed_sentences = []
    for tokens in tokenized_sentences:
        indexed_sentences.append([glove_stoi.get(token, glove_stoi.get('<unk>', 0)) for token in tokens])

    # Pad sequences and get lengths for the RNN
    # Find the maximum length among all tokenized sentences
    max_len = max(len(s) for s in indexed_sentences) if indexed_sentences else 0
    if max_len == 0: # Handle case where all sentences are empty after tokenization
        return issues

    padded_indexed_sentences = []
    lengths = []
    for s in indexed_sentences:
        padded_indexed_sentences.append(s + [0] * (max_len - len(s))) # Pad with 0s
        lengths.append(len(s))

    # Convert to tensors
    # Ensure lengths_tensor is on CPU for pack_padded_sequence
    input_tensor = torch.tensor(padded_indexed_sentences, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    # Get embeddings from the encoder
    with torch.no_grad(): # No need to calculate gradients for inference
        embeddings = encoder(input_tensor, lengths_tensor)

    # Calculate cosine similarity between all pairs of embeddings
    # Normalize embeddings for cosine similarity calculation
    norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    # Transpose the second tensor for dot product to get similarity matrix
    similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.transpose(0, 1))

    # Check for high similarity (redundancy)
    # You can adjust this threshold based on how strict you want the redundancy check to be
    redundancy_threshold = 0.80 
    
    for i in range(len(sentences)):
        # Start j from i + 1 to avoid comparing a sentence with itself and to avoid duplicate pairs
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i, j] > redundancy_threshold:
                issues.append({
                    "issue_type": "Redundancy",
                    "description": f"Sentences are highly similar (similarity: {similarity_matrix[i, j]:.2f}): '{sentences[i]}' and '{sentences[j]}'",
                    "suggested_fix": "Consider rephrasing or combining these sentences to avoid repetition and improve clarity."
                })
    return issues

# You can keep your old keyword-based check if you want it as a separate, simpler check,
# but it won't be used by the validator's current setup for 'check_redundancy'.
# If you want to use it, you'd need to add it as another check in get_validation_checks()
# and call it appropriately.
def check_redundancy_keywords(prompt: str) -> List[dict]:
    """
    Checks for redundant instructions by analyzing repeating keywords and phrases.
    This is a simpler, keyword-based check.
    """
    issues = []
    normalized_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    words = normalized_prompt.split()
    
    if len(words) < 5: # Not enough words to meaningfully check for keyword redundancy
        return issues
        
    instructional_keywords = ["detailed", "descriptive", "comprehensive", "very", "extremely", "many", "always", "ensure", "must"]
    keyword_counts = Counter(word for word in words if word in instructional_keywords)
    
    for keyword, count in keyword_counts.items():
        if count > 1:
            issues.append({
                "issue_type": "Keyword Redundancy",
                "description": f"The keyword '{keyword}' appears multiple times ({count} times), suggesting potential redundant emphasis.",
                "suggested_fix": f"Review the usage of '{keyword}' to ensure it's not overused or redundant."
            })
    return issues