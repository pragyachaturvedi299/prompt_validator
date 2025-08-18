import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import numpy as np

# Load pre-trained GloVe embeddings
# Download with: python -c "import torchtext; torchtext.vocab.GloVe(name='6B', dim=100);"
glove = GloVe(name='6B', dim=100)

class RedundancyClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# A simple tokenizer to convert sentences to sequences of integers
def tokenize_sentence(sentence, vocab):
    return [vocab[token.lower()] for token in sentence.split() if token.lower() in vocab]

# A placeholder function to load the pre-trained model. In a real scenario, you'd train this model
# on a dataset of redundant/non-redundant sentence pairs.
def load_redundancy_model():
    VOCAB_SIZE = len(glove.stoi)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    OUTPUT_DIM = 2 # e.g., [0] for not redundant, [1] for redundant
    DROPOUT_RATE = 0.5
    
    model = RedundancyClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT_RATE)
    # Here you would load state_dict if you had a trained model
    # model.load_state_dict(torch.load('redundancy_model.pt'))
    model.eval()
    return model