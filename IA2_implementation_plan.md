# ICT4403 – Foundations of Generative AI | Assignment 2
## Autoregressive Text Generation System — Implementation Plan

---

## 📋 Overview

Build a **word-level autoregressive text generation system** that trains and compares four recurrent architectures — **Simple RNN, Bidirectional RNN, LSTM, and GRU** — on the same dataset, then analyzes temperature-based sampling.

---

## Phase 1 — Dataset Selection & Preprocessing

### 1.1 Dataset Choice (Recommended Options)
| Option | Description | Why It Works |
|---|---|---|
| **Shakespeare's Works** (TensorFlow dataset) | ~1MB, rich literary text | Classic benchmark, good coherence to measure |
| **Tiny Shakespeare** (Andrej Karpathy's subset) | ~1MB prose | Small, fast to train |
| **NLTK Gutenberg corpus** (e.g., *Emma* by Austen) | ~880KB fiction | Clean, English prose |
| **WikiText-2** | ~2MB Wikipedia articles | Structured, diverse vocabulary |

> **Recommendation:** Use **Tiny Shakespeare** — well-known, royalty-free, easy to source, and yields interesting generated text that is easy to evaluate qualitatively.
> Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

### 1.2 Dataset Characteristics to Describe
- Total characters / words / sentences
- Vocabulary size (unique tokens)
- Average sentence length
- Distribution of word frequencies (zipf plot)
- Any domain-specific features (e.g., archaic language, dialogue tags)

### 1.3 Preprocessing Pipeline
```
Raw Text
  │
  ▼
Lowercase + Remove special chars (keep punctuation)
  │
  ▼
Tokenise (word-level using Keras Tokenizer or torchtext)
  │
  ▼
Build vocabulary (map word → integer index)
  │
  ▼
Create sequences: sliding window of length N (e.g., N=10)
  - Input  : words[0 : N-1]
  - Target : words[N]  (next word)
  │
  ▼
Encode sequences → integer arrays
  │
  ▼
One-hot / embedding lookup (embedding layer inside model)
  │
  ▼
Train / Validation split (80/20)
```

**Key parameters to note:**
- `SEQUENCE_LENGTH` = 10 (tunable)
- `VOCAB_SIZE` = top-K words (e.g., 10,000)
- `EMBEDDING_DIM` = 128

---

## Phase 2 — Model Implementation

All four models share the **same hyperparameters** for a fair comparison:

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 128 |
| Recurrent units | 128 |
| Dropout | 0.2 |
| Dense output | Softmax over vocab |
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse categorical crossentropy |
| Epochs | 20 |
| Batch size | 128 |

### 2.1 Simple RNN
```python
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQ_LEN),
    SimpleRNN(128, dropout=0.2),
    Dense(VOCAB_SIZE, activation='softmax')
])
```

### 2.2 Bidirectional RNN
```python
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQ_LEN),
    Bidirectional(SimpleRNN(128, dropout=0.2)),
    Dense(VOCAB_SIZE, activation='softmax')
])
```

### 2.3 LSTM
```python
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQ_LEN),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(VOCAB_SIZE, activation='softmax')
])
```

### 2.4 GRU
```python
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQ_LEN),
    GRU(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(VOCAB_SIZE, activation='softmax')
])
```

> **Note:** All models are compiled, summarised, and trained identically so experimental conditions are comparable.

---

## Phase 3 — Evaluation & Comparison

### 3.1 Quantitative Metrics

| Metric | How to Compute |
|---|---|
| **Training Loss** | Recorded per epoch (cross-entropy) |
| **Validation Loss** | Recorded per epoch on held-out set |
| **Perplexity** | `exp(val_loss)` — lower is better |
| **Prediction Accuracy** | Top-1 accuracy on next-word prediction on validation set |
| **Training Time (s)** | Measure wall-clock time per epoch using `time` module |

### 3.2 Qualitative Metrics

| Metric | How to Assess |
|---|---|
| **Coherence** | Do generated sentences make grammatical/semantic sense? |
| **Diversity** | Variety of words/phrases across different seed texts |
| **Contextual Consistency** | Does the text stay on-topic? |
| **Repetition** | Count repeated n-grams; lower is better |

### 3.3 Comparison Table (Fill in after training)
| Model | Val Loss | Perplexity | Top-1 Accuracy | Train Time/Epoch | Coherence | Repetition |
|---|---|---|---|---|---|---|
| Simple RNN | — | — | — | — | — | — |
| BiRNN | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — |
| GRU | — | — | — | — | — | — |

### 3.4 Plots to Generate
- Training & validation loss curves (all 4 models on same plot)
- Perplexity over epochs
- Bar chart: training time per model
- Bar chart: accuracy per model

---

## Phase 4 — Text Generation with Temperature Sampling

### 4.1 Seed Sequences (use at least 2–3)
- e.g., `"to be or not to be that is"`
- e.g., `"all the world is a stage and"`
- e.g., `"shall i compare thee to a summer"`

### 4.2 Temperature Values to Test
- **T = 0.2** → Low (conservative, repetitive, high confidence)
- **T = 0.5** → Medium (balanced)
- **T = 1.0** → Neutral (default softmax distribution)
- **T = 1.5** → High (creative but potentially incoherent)

### 4.3 Sampling Function
```python
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)
```

### 4.4 Analysis Matrix
| Model | T=0.2 | T=0.5 | T=1.0 | T=1.5 |
|---|---|---|---|---|
| Simple RNN | _generated text_ | _generated text_ | _generated text_ | _generated text_ |
| BiRNN | — | — | — | — |
| LSTM | — | — | — | — |
| GRU | — | — | — | — |

**Justify:** Which model + temperature combination gives the best trade-off between coherence and diversity?

---

## Phase 5 — Repository Setup (GitHub)

### 5.1 Recommended File Structure
```
autoregressive-text-gen/
│
├── data/
│   └── dataset_source_link.txt          # or dataset file if small
│
├── notebooks/
│   └── IA2_text_generation.ipynb        # main Jupyter notebook
│
├── src/                                  # optional, if using .py files
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── results/
│   ├── loss_curves.png
│   ├── perplexity_comparison.png
│   ├── accuracy_comparison.png
│   ├── training_time_comparison.png
│   └── generated_text_samples.txt
│
└── README.md
```

### 5.2 README.md Sections (required by assignment)
1. Problem statement and objective
2. Dataset description and source link
3. Brief description of each model
4. Execution steps and dependencies (`requirements.txt`)
5. Results obtained (tables + plots embedded)
6. Model comparison and evaluation
7. Sampling-temperature analysis
8. Conclusion — best-performing model with justification

---

## Phase 6 — Submission Checklist

- [ ] Jupyter Notebook runs end-to-end without errors
- [ ] All 4 models implemented with identical hyperparameters
- [ ] Quantitative evaluation table complete
- [ ] At least 3 temperature values tested with generated outputs
- [ ] Plots saved and included in results/
- [ ] GitHub repo created (public or evaluator has access)
- [ ] README.md contains all required sections
- [ ] Dataset source link included
- [ ] Word document created: `YourName_RegNum_FGAI_Assignment2.docx`
- [ ] Word doc converted to PDF
- [ ] PDF uploaded to LMS before deadline

---

## Suggested Timeline

| Day | Task |
|---|---|
| Day 1 | Dataset download, preprocessing, vocabulary building, sequence creation |
| Day 2 | Implement & train all 4 models; save checkpoints |
| Day 3 | Evaluate metrics, generate plots, run temperature sampling |
| Day 4 | Write README, clean up code, push to GitHub |
| Day 5 | Create submission document, convert to PDF, upload to LMS |

---

## Dependencies (`requirements.txt`)
```
tensorflow>=2.12
numpy
pandas
matplotlib
scikit-learn
nltk
jupyter
```

> [!TIP]
> Use **Google Colab** with a T4 GPU runtime for faster training — free and requires no local setup.

> [!IMPORTANT]
> All four models must be trained under **identical experimental conditions** (same dataset split, same hyperparameters, same number of epochs) to ensure a valid comparison.
