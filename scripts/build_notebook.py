import json
import os

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Foundations of Generative AI (ICT4403) — Assignment 2\n",
                    "## Autoregressive Text Generation System\n",
                    "### Comparative Analysis of Simple RNN, Bidirectional RNN, LSTM, and GRU\n",
                    "\n",
                    "---\n",
                    "**Objective:**\n",
                    "1. Preprocess and analyze a text corpus (Tiny Shakespeare) and construct sliding-window input-target sequences for next-word prediction.\n",
                    "2. Implement and train four recurrent architectures (**Simple RNN, Bidirectional RNN, LSTM, GRU**) under strictly comparable experimental conditions.\n",
                    "3. Evaluate the models using quantitative metrics (**Training/Validation Loss, Perplexity, Top-1 Accuracy, Execution Time**) and qualitative characteristics (**Coherence, Diversity, Repetition**).\n",
                    "4. Autoregressively generate text from seed prompts across multiple sampling temperatures ($T = 0.2, 0.5, 1.0, 1.5$) to analyze the trade-off between randomness and fluency."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Hardware Acceleration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import re\n",
                    "import time\n",
                    "import math\n",
                    "import json\n",
                    "from collections import Counter\n",
                    "\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.nn.functional as F\n",
                    "from torch.utils.data import Dataset, DataLoader, random_split\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Set random seed for strict reproducibility\n",
                    "SEED = 42\n",
                    "torch.manual_seed(SEED)\n",
                    "np.random.seed(SEED)\n",
                    "\n",
                    "# Auto-detect device: MPS (Apple Silicon), CUDA, or CPU\n",
                    "if torch.backends.mps.is_available():\n",
                    "    device = torch.device('mps')\n",
                    "elif torch.cuda.is_available():\n",
                    "    device = torch.device('cuda')\n",
                    "else:\n",
                    "    device = torch.device('cpu')\n",
                    "print(f\"Active compute device: {device}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Acquisition & Characterization\n",
                    "We utilize the **Tiny Shakespeare** dataset (~1.11 MB, ~40,000 lines, 202,651 words) from Andrej Karpathy's benchmark."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "DATA_PATH = '../data/tinyshakespeare.txt' if os.path.exists('../data/tinyshakespeare.txt') else 'data/tinyshakespeare.txt'\n",
                    "\n",
                    "with open(DATA_PATH, 'r', encoding='utf-8') as f:\n",
                    "    raw_text = f.read()\n",
                    "\n",
                    "print(f\"Raw text length: {len(raw_text):,} characters\")\n",
                    "print(f\"Total lines: {len(raw_text.splitlines()):,}\")\n",
                    "print(\"\\n--- First 300 characters sample ---\")\n",
                    "print(raw_text[:300])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.1 Word-Level Tokenization & Vocabulary Construction"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract words and preserve basic punctuation tokens\n",
                    "tokens = re.findall(r\"\\b\\w+(?:'\\w+)?|[.,!?:;]\", raw_text.lower())\n",
                    "print(f\"Total extracted word tokens: {len(tokens):,}\")\n",
                    "\n",
                    "class Vocabulary:\n",
                    "    def __init__(self, pad_token='<pad>', unk_token='<unk>'):\n",
                    "        self.pad_token = pad_token\n",
                    "        self.unk_token = unk_token\n",
                    "        self.word2idx = {}\n",
                    "        self.idx2word = {}\n",
                    "        self.pad_idx = self.add_word(self.pad_token)\n",
                    "        self.unk_idx = self.add_word(self.unk_token)\n",
                    "\n",
                    "    def add_word(self, word):\n",
                    "        if word not in self.word2idx:\n",
                    "            idx = len(self.word2idx)\n",
                    "            self.word2idx[word] = idx\n",
                    "            self.idx2word[idx] = word\n",
                    "            return idx\n",
                    "        return self.word2idx[word]\n",
                    "\n",
                    "    def build_from_words(self, words, min_freq=2, max_vocab_size=5000):\n",
                    "        counter = Counter(words)\n",
                    "        most_common = counter.most_common(max_vocab_size - len(self.word2idx))\n",
                    "        for word, count in most_common:\n",
                    "            if count >= min_freq:\n",
                    "                self.add_word(word)\n",
                    "\n",
                    "    def __len__(self):\n",
                    "        return len(self.word2idx)\n",
                    "\n",
                    "    def encode(self, words):\n",
                    "        return [self.word2idx.get(w, self.unk_idx) for w in words]\n",
                    "\n",
                    "    def decode(self, indices):\n",
                    "        return [self.idx2word.get(idx, self.unk_token) for idx in indices]\n",
                    "\n",
                    "vocab = Vocabulary()\n",
                    "vocab.build_from_words(tokens, min_freq=2, max_vocab_size=5000)\n",
                    "print(f\"Vocabulary Size: {len(vocab):,} unique words (including <pad>, <unk>)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.2 Zipf's Law Word Frequency Distribution Plot"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "counter = Counter(tokens)\n",
                    "top_30 = counter.most_common(30)\n",
                    "words, counts = zip(*top_30)\n",
                    "\n",
                    "plt.figure(figsize=(12, 4))\n",
                    "plt.bar(words, counts, color='#2b5c8f')\n",
                    "plt.xticks(rotation=60, ha='right')\n",
                    "plt.title(\"Top 30 Most Frequent Words in Shakespeare Corpus\")\n",
                    "plt.ylabel(\"Frequency\")\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.3 Sliding-Window Autoregressive Sequence Generation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "SEQ_LENGTH = 8\n",
                    "BATCH_SIZE = 128\n",
                    "SAMPLE_LIMIT = 60000  # Subset for fast & reproducible training\n",
                    "\n",
                    "encoded_tokens = vocab.encode(tokens[:SAMPLE_LIMIT])\n",
                    "\n",
                    "class TextSequenceDataset(Dataset):\n",
                    "    def __init__(self, token_indices, seq_length=8):\n",
                    "        self.inputs = []\n",
                    "        self.targets = []\n",
                    "        for i in range(len(token_indices) - seq_length):\n",
                    "            self.inputs.append(token_indices[i : i + seq_length])\n",
                    "            self.targets.append(token_indices[i + seq_length])\n",
                    "        self.inputs = torch.tensor(self.inputs, dtype=torch.long)\n",
                    "        self.targets = torch.tensor(self.targets, dtype=torch.long)\n",
                    "\n",
                    "    def __len__(self):\n",
                    "        return len(self.targets)\n",
                    "\n",
                    "    def __getitem__(self, idx):\n",
                    "        return self.inputs[idx], self.targets[idx]\n",
                    "\n",
                    "full_dataset = TextSequenceDataset(encoded_tokens, seq_length=SEQ_LENGTH)\n",
                    "val_size = int(len(full_dataset) * 0.2)\n",
                    "train_size = len(full_dataset) - val_size\n",
                    "\n",
                    "gen = torch.Generator().manual_seed(SEED)\n",
                    "train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=gen)\n",
                    "\n",
                    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)\n",
                    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
                    "\n",
                    "print(f\"Total Sequences : {len(full_dataset):,}\")\n",
                    "print(f\"Training Batches: {len(train_loader)} (Size: {BATCH_SIZE})\")\n",
                    "print(f\"Validation Batches: {len(val_loader)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Model Architecture Implementations\n",
                    "We implement 4 recurrent variants sharing the exact same hyperparameters:\n",
                    "- **Embedding Dimension**: 128\n",
                    "- **Hidden Dimension**: 128\n",
                    "- **Dropout**: 0.2\n",
                    "- **Optimization**: Adam (lr=0.001)\n",
                    "- **Loss Function**: Cross-Entropy Loss"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class SimpleRNNLM(nn.Module):\n",
                    "    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):\n",
                    "        super().__init__()\n",
                    "        self.name = \"Simple RNN\"\n",
                    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
                    "        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)\n",
                    "        self.dropout = nn.Dropout(dropout)\n",
                    "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
                    "\n",
                    "    def forward(self, x):\n",
                    "        embeds = self.embedding(x)\n",
                    "        out, _ = self.rnn(embeds)\n",
                    "        return self.fc(self.dropout(out[:, -1, :]))\n",
                    "\n",
                    "class BiRNNLM(nn.Module):\n",
                    "    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):\n",
                    "        super().__init__()\n",
                    "        self.name = \"Bidirectional RNN\"\n",
                    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
                    "        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)\n",
                    "        self.dropout = nn.Dropout(dropout)\n",
                    "        self.fc = nn.Linear(hidden_dim * 2, vocab_size)\n",
                    "\n",
                    "    def forward(self, x):\n",
                    "        embeds = self.embedding(x)\n",
                    "        out, _ = self.rnn(embeds)\n",
                    "        return self.fc(self.dropout(out[:, -1, :]))\n",
                    "\n",
                    "class LSTMLM(nn.Module):\n",
                    "    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):\n",
                    "        super().__init__()\n",
                    "        self.name = \"LSTM\"\n",
                    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
                    "        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)\n",
                    "        self.dropout = nn.Dropout(dropout)\n",
                    "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
                    "\n",
                    "    def forward(self, x):\n",
                    "        embeds = self.embedding(x)\n",
                    "        out, _ = self.lstm(embeds)\n",
                    "        return self.fc(self.dropout(out[:, -1, :]))\n",
                    "\n",
                    "class GRULM(nn.Module):\n",
                    "    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.2):\n",
                    "        super().__init__()\n",
                    "        self.name = \"GRU\"\n",
                    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
                    "        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)\n",
                    "        self.dropout = nn.Dropout(dropout)\n",
                    "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
                    "\n",
                    "    def forward(self, x):\n",
                    "        embeds = self.embedding(x)\n",
                    "        out, _ = self.gru(embeds)\n",
                    "        return self.fc(self.dropout(out[:, -1, :]))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Training Engine & Comparable Evaluation Loop"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device):\n",
                    "    model = model.to(device)\n",
                    "    criterion = nn.CrossEntropyLoss()\n",
                    "    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
                    "    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_ppl': [], 'epoch_times': []}\n",
                    "    \n",
                    "    print(f\"\\nTraining {model.name}...\")\n",
                    "    for epoch in range(1, epochs + 1):\n",
                    "        start = time.time()\n",
                    "        model.train()\n",
                    "        train_loss, train_correct, train_total = 0.0, 0, 0\n",
                    "        for x_b, y_b in train_loader:\n",
                    "            x_b, y_b = x_b.to(device), y_b.to(device)\n",
                    "            optimizer.zero_grad()\n",
                    "            logits = model(x_b)\n",
                    "            loss = criterion(logits, y_b)\n",
                    "            loss.backward()\n",
                    "            nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                    "            optimizer.step()\n",
                    "            train_loss += loss.item() * x_b.size(0)\n",
                    "            train_total += y_b.size(0)\n",
                    "            \n",
                    "        # Validation\n",
                    "        model.eval()\n",
                    "        val_loss, val_correct, val_total = 0.0, 0, 0\n",
                    "        with torch.no_grad():\n",
                    "            for x_b, y_b in val_loader:\n",
                    "                x_b, y_b = x_b.to(device), y_b.to(device)\n",
                    "                logits = model(x_b)\n",
                    "                loss = criterion(logits, y_b)\n",
                    "                val_loss += loss.item() * x_b.size(0)\n",
                    "                preds = torch.argmax(logits, dim=-1)\n",
                    "                val_correct += (preds == y_b).sum().item()\n",
                    "                val_total += y_b.size(0)\n",
                    "                \n",
                    "        t_loss = train_loss / train_total\n",
                    "        v_loss = val_loss / val_total\n",
                    "        v_acc = val_correct / val_total\n",
                    "        v_ppl = math.exp(min(v_loss, 20))\n",
                    "        duration = time.time() - start\n",
                    "        \n",
                    "        history['train_loss'].append(t_loss)\n",
                    "        history['val_loss'].append(v_loss)\n",
                    "        history['val_acc'].append(v_acc)\n",
                    "        history['val_ppl'].append(v_ppl)\n",
                    "        history['epoch_times'].append(duration)\n",
                    "        print(f\"Epoch {epoch:2d}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:7.2f} | Val Acc: {v_acc*100:5.2f}% | Time: {duration:.2f}s\")\n",
                    "    return history"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Comparative Evaluation Results & Graphs"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load precomputed results or run live\n",
                    "RESULTS_JSON = '../results/metrics_history.json' if os.path.exists('../results/metrics_history.json') else 'results/metrics_history.json'\n",
                    "\n",
                    "if os.path.exists(RESULTS_JSON):\n",
                    "    with open(RESULTS_JSON, 'r') as f:\n",
                    "        results = json.load(f)\n",
                    "    print(\"Loaded experiment history successfully!\")\n",
                    "else:\n",
                    "    print(\"Run 'python3 run_experiments.py' to generate history, or train models directly.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5.1 Learning Curves (Train & Validation Loss)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'results' in locals():\n",
                    "    plt.figure(figsize=(12, 5))\n",
                    "    for name, hist in results.items():\n",
                    "        plt.plot(hist['val_loss'], label=f\"{name} (Val Loss)\", linewidth=2)\n",
                    "    plt.title(\"Validation Loss Comparison Across Recurrent Architectures\")\n",
                    "    plt.xlabel(\"Epoch\")\n",
                    "    plt.ylabel(\"Loss\")\n",
                    "    plt.grid(True, linestyle='--', alpha=0.5)\n",
                    "    plt.legend()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Autoregressive Text Generation Across Temperatures\n",
                    "We test $T = 0.2, 0.5, 1.0, 1.5$ to analyze coherence vs diversity."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def sample_with_temperature(logits, temperature=1.0):\n",
                    "    if temperature <= 1e-4:\n",
                    "        return torch.argmax(logits, dim=-1).item()\n",
                    "    probs = F.softmax(logits / temperature, dim=-1)\n",
                    "    return torch.multinomial(probs, 1).item()\n",
                    "\n",
                    "def generate(model, seed_text, max_tokens=30, temperature=1.0, seq_len=8):\n",
                    "    model.eval()\n",
                    "    words = re.findall(r\"\\b\\w+(?:'\\w+)?|[.,!?:;]\", seed_text.lower())\n",
                    "    gen_words = list(words)\n",
                    "    with torch.no_grad():\n",
                    "        for _ in range(max_tokens):\n",
                    "            ctx = gen_words[-seq_len:]\n",
                    "            enc = vocab.encode(ctx)\n",
                    "            if len(enc) < seq_len:\n",
                    "                enc = [vocab.pad_idx] * (seq_len - len(enc)) + enc\n",
                    "            inp = torch.tensor([enc], dtype=torch.long).to(device)\n",
                    "            logits = model(inp).squeeze(0)\n",
                    "            logits[vocab.pad_idx] = -float('inf')\n",
                    "            logits[vocab.unk_idx] = -float('inf')\n",
                    "            next_idx = sample_with_temperature(logits, temperature)\n",
                    "            gen_words.append(vocab.idx_to_word(next_idx))\n",
                    "    return ' '.join(gen_words)\n",
                    "\n",
                    "print(\"Generation function ready.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/text_generation_assignment2.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/text_generation_assignment2.ipynb")

if __name__ == "__main__":
    create_notebook()
