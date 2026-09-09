"""Model definitions and shared training configuration.

The transformer settings are the ones used for the reference models and are held
fixed across every arm so that comparisons isolate the quantity under study.
"""
import torch
import torch.nn as nn

# transformer tiers
HF_NAMES = {'muril': 'google/muril-base-cased', 'xlmr': 'xlm-roberta-base'}
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
MAX_EPOCHS = 6
PATIENCE = 2

# tier-1 gate
TFIDF_KWARGS = dict(ngram_range=(1, 2), max_features=200_000, min_df=2)
LOGREG_KWARGS = dict(max_iter=3000, C=5)

# recurrent tier-1 alternative
LSTM_MAX_LEN = 256
EMB_DIM = 300
HIDDEN = 256
N_LAYERS = 2
DROPOUT = 0.5
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 32
VOCAB_SIZE = 40_000
PAD, UNK = 0, 1


class BiLSTM(nn.Module):
    """Two-layer bidirectional LSTM over word embeddings, 14,720,770 parameters
    at a vocabulary of 40,000."""

    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, ids, lengths):
        embedded = self.drop_in(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.drop_out(pooled))
