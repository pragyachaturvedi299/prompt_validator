import torchtext
from torchtext.vocab import GloVe

def get_glove_embeddings(dim=100):
    """
    Loads pre-trained GloVe embeddings.

    Args:
        dim (int): The dimensionality of the word vectors. Default is 100.

    Returns:
        torchtext.vocab.Vocab: A GloVe object containing word vectors and a vocabulary.
    """
    print(f"Loading GloVe embeddings with {dim} dimensions...")
    try:
        glove = GloVe(name='6B', dim=dim)
        print("GloVe embeddings loaded successfully.")
        return glove
    except RuntimeError:
        print("Failed to load GloVe embeddings. Please download them first.")
        print("You can download them by running 'python -m torchtext.datasets.glove' or similar.")
        return None