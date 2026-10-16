import torch
import torch.nn as nn


class SimpleRNNLanguageModel(nn.Module):
    """
    Autoregressive Next-Word Prediction with Simple (Vanilla) RNN.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):
        super(SimpleRNNLanguageModel, self).__init__()
        self.name = "Simple RNN"
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)                     # (batch_size, seq_len, embed_dim)
        out, h_n = self.rnn(embeds)                    # out: (batch_size, seq_len, hidden_dim)
        last_step = out[:, -1, :]                      # representation at final token
        dropped = self.dropout(last_step)
        logits = self.fc(dropped)                      # (batch_size, vocab_size)
        return logits


class BiRNNLanguageModel(nn.Module):
    """
    Autoregressive Next-Word Prediction with Bidirectional RNN.
    Encodes the given seed sequence in both forward and backward directions.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):
        super(BiRNNLanguageModel, self).__init__()
        self.name = "Bidirectional RNN"
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Bidirectional concatenates forward and backward hidden features: 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        out, h_n = self.rnn(embeds)                    # out: (batch_size, seq_len, hidden_dim * 2)
        # out[:, -1, :hidden_dim] is forward last step
        # out[:, -1, hidden_dim:] or out[:, 0, hidden_dim:] captures backward pass
        # Taking out[:, -1, :] provides the combined representation at the boundary
        last_step = out[:, -1, :]
        dropped = self.dropout(last_step)
        logits = self.fc(dropped)                      # (batch_size, vocab_size)
        return logits


class LSTMLanguageModel(nn.Module):
    """
    Autoregressive Next-Word Prediction with Long Short-Term Memory (LSTM).
    Leverages gating mechanisms (input, forget, output gates) to mitigate vanishing gradients.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        self.name = "LSTM"
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embeds)            # out: (batch_size, seq_len, hidden_dim)
        last_step = out[:, -1, :]
        dropped = self.dropout(last_step)
        logits = self.fc(dropped)                      # (batch_size, vocab_size)
        return logits


class GRULanguageModel(nn.Module):
    """
    Autoregressive Next-Word Prediction with Gated Recurrent Unit (GRU).
    Uses reset and update gates for efficient dependency tracking with fewer parameters than LSTM.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):
        super(GRULanguageModel, self).__init__()
        self.name = "GRU"
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        out, h_n = self.gru(embeds)                    # out: (batch_size, seq_len, hidden_dim)
        last_step = out[:, -1, :]
        dropped = self.dropout(last_step)
        logits = self.fc(dropped)                      # (batch_size, vocab_size)
        return logits


def get_model(model_name, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):
    """Helper factory for all 4 models."""
    key = model_name.lower().replace(" ", "").replace("_", "")
    if "birnn" in key or "bidirectional" in key:
        return BiRNNLanguageModel(vocab_size, embedding_dim, hidden_dim, dropout)
    elif "simplernn" in key or key == "rnn":
        return SimpleRNNLanguageModel(vocab_size, embedding_dim, hidden_dim, dropout)
    elif "lstm" in key:
        return LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, dropout)
    elif "gru" in key:
        return GRULanguageModel(vocab_size, embedding_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
