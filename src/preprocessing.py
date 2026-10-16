import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class Vocabulary:
    """Vocabulary mapping for word-level tokenization."""
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        
        # Reserved tokens
        self.pad_idx = self.add_word(self.pad_token)
        self.unk_idx = self.add_word(self.unk_token)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def build_from_words(self, words, min_freq=2, max_vocab_size=6000):
        counter = Counter(words)
        # Sort words by frequency
        most_common = counter.most_common(max_vocab_size - len(self.word2idx))
        for word, count in most_common:
            if count >= min_freq:
                self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def word_to_idx(self, word):
        return self.word2idx.get(word, self.unk_idx)

    def idx_to_word(self, idx):
        return self.idx2word.get(idx, self.unk_token)

    def encode(self, words):
        return [self.word_to_idx(w) for w in words]

    def decode(self, indices):
        return [self.idx_to_word(idx) for idx in indices]


def clean_and_tokenize(text):
    """
    Cleans text and extracts word-level tokens while preserving essential punctuation
    as distinct tokens to maintain poetic and dialogue structure.
    """
    # Replace newlines with spaces or keep formatting tokens
    text = text.lower()
    # Separate punctuation characters with spaces so they become distinct tokens or retain word characters
    # Using regex to split words and punctuation
    tokens = re.findall(r"\b\w+(?:'\w+)?|[.,!?:;]", text)
    return tokens


def get_dataset_stats(raw_text, tokens, vocab):
    """
    Computes statistical characteristics of the textual dataset.
    """
    lines = raw_text.splitlines()
    total_chars = len(raw_text)
    total_words = len(tokens)
    total_lines = len(lines)
    unique_words = len(set(tokens))
    vocab_size = len(vocab)
    avg_words_per_line = total_words / max(1, total_lines)
    
    # Calculate word frequency distribution for Zipf analysis
    counter = Counter(tokens)
    top_10 = counter.most_common(10)
    
    stats = {
        "total_characters": total_chars,
        "total_tokens": total_words,
        "total_lines": total_lines,
        "unique_raw_tokens": unique_words,
        "vocab_size": vocab_size,
        "avg_words_per_line": avg_words_per_line,
        "top_10_words": top_10
    }
    return stats


class TextSequenceDataset(Dataset):
    """
    Autoregressive sliding-window dataset:
    Given a sequence of length N: [w_1, w_2, ..., w_N],
    predict target next word w_{N+1}.
    """
    def __init__(self, token_indices, seq_length=8):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        
        # Build sliding windows
        for i in range(len(token_indices) - seq_length):
            self.inputs.append(token_indices[i : i + seq_length])
            self.targets.append(token_indices[i + seq_length])
            
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def prepare_data(file_path, seq_length=8, max_vocab_size=5000, min_freq=2, val_split=0.2, batch_size=128, max_samples=None, seed=42):
    """
    Full pipeline to load, preprocess, tokenize, and create train/val DataLoaders.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = clean_and_tokenize(raw_text)
    
    vocab = Vocabulary()
    vocab.build_from_words(tokens, min_freq=min_freq, max_vocab_size=max_vocab_size)
    
    token_indices = vocab.encode(tokens)
    
    if max_samples is not None and max_samples < len(token_indices):
        token_indices = token_indices[:max_samples]
        
    dataset = TextSequenceDataset(token_indices, seq_length=seq_length)
    
    # Train / Val Split
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    stats = get_dataset_stats(raw_text, tokens, vocab)
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "vocab": vocab,
        "stats": stats,
        "tokens": tokens,
        "raw_text": raw_text
    }
