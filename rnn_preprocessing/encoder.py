# In rnn_preprocessing/encoder.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, glove_vectors):
        super(SentenceEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(glove_vectors, freeze=True)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, token_ids, lengths):
        # Embed the tokens
        embedded_tokens = self.embedding(token_ids)
        
        # Pack the sequences for the RNN
        packed_tokens = pack_padded_sequence(embedded_tokens, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass packed sequences to the RNN
        _, hidden = self.rnn(packed_tokens)
        
        # Return the hidden state of the last time step
        return hidden.squeeze(0)