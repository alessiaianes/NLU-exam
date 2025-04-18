import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_LSTM_weight_tying(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_weight_tying, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: Condividi i pesi tra l'embedding e l'output layer
        self.output.weight = self.embedding.weight  # Condivisione dei pesi
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    

class VariationalDropout(nn.Module):
    """
    Implement Variational Dropout that keeps the same dropout mask for all timesteps.
    """
    def __init__(self, dropout_prob):
        super(VariationalDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if not self.training or self.dropout_prob == 0:
            return x
        
        # Create a dropout mask for the entire sequence
        mask = x.new_empty(x.size(0), x.size(2)).bernoulli_(1 - self.dropout_prob)
        mask = mask.div_(1 - self.dropout_prob)  # Invert the probability
        mask = mask.unsqueeze(1)  # Expand dimension for broadcasting

        return x * mask


class LM_LSTM_wt_vd(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, 
                 emb_dropout=0.1, out_dropout=0.4):
        super(LM_LSTM_wt_vd, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Variational Dropout layers
        self.embedding_dropout = VariationalDropout(emb_dropout)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.output_dropout = VariationalDropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: Shared weights between embedding and output layer
        self.output.weight = self.embedding.weight
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        # Apply embedding dropout
        emb = self.embedding(input_sequence)
        emb = self.embedding_dropout(emb)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(emb)
        
        # Apply output dropout
        lstm_out = self.output_dropout(lstm_out)
        
        # Output layer
        output = self.output(lstm_out).permute(0, 2, 1)
        return output