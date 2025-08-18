import re

def chunk_into_sentences(text):
    """
    Splits a given text into a list of sentences.

    This function uses a simple regular expression to identify sentence
    boundaries. It handles common cases like periods, question marks,
    and exclamation points, while trying to ignore abbreviations.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of strings, where each string is a sentence.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]