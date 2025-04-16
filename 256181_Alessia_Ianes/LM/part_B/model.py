import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_LSTM_wt_vd(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.1):
        super(LM_LSTM_wt_vd, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        # Dropout layers for variational dropout
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: Condividi i pesi tra l'embedding e l'output layer
        self.output.weight = self.embedding.weight  # Condivisione dei pesi
        self.pad_token = pad_index
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout(emb)  # Apply the same dropout mask to all tokens in the sequence

          # LSTM + Variational Dropout
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)  # Apply the same dropout mask to all tokens in the sequence
        

        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    