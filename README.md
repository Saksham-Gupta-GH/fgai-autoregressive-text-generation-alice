# Foundations of Generative AI (ICT4403) — Assignment 2
## Autoregressive Text Generation System: Comparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU

---

## 1. Problem Statement & Objectives

Autoregressive language models predict each succeeding token conditioned on all previously generated tokens:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

The objectives of this assignment are:
1. **Data Preparation**: Preprocess a textual corpus (**Tiny Shakespeare**), construct a word-level vocabulary, and create sliding-window input–target sequences $([w_{t-k}, \dots, w_{t-1}] \to w_t)$.
2. **Model Implementation**: Build four recurrent neural architectures from scratch:
   - **Simple RNN** (Vanilla Elman RNN)
   - **Bidirectional RNN** (BiRNN)
   - **LSTM** (Long Short-Term Memory)
   - **GRU** (Gated Recurrent Unit)
3. **Rigorous Comparison**: Train all four models under **identical, comparable experimental conditions** (same seed, learning rate, loss function, optimizer, batch size, embedding dimension, and hidden units).
4. **Quantitative & Qualitative Evaluation**: Compare models using:
   - Quantitative: Cross-Entropy Loss, Perplexity ($\exp(\mathcal{L})$), Top-1 Accuracy, Training Duration.
   - Qualitative: Syntactic coherence, contextual consistency, Distinct-1 / Distinct-2 diversity, and n-gram repetition.
5. **Temperature Analysis**: Autoregressively generate text from seed prompts across four sampling temperatures ($T = 0.2, 0.5, 1.0, 1.5$) to evaluate the trade-off between randomness, repetition, and fluency.

---

## 2. Dataset Description & Source Link

- **Dataset Name**: Tiny Shakespeare
- **Source Link**: [https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- **Origin**: Curated by Andrej Karpathy for language modeling benchmarks, sourced from Project Gutenberg.

### Dataset Statistical Characteristics
| Metric | Value |
| :--- | :--- |
| **Total Characters** | 1,115,394 |
| **Total Word Tokens** | 250,175 |
| **Total Lines** | 40,000 |
| **Vocabulary Size** | 5,000 unique words (with `<pad>` and `<unk>`) |
| **Average Words per Line** | ~6.25 words |
| **Top-5 Frequent Words** | `the` (5,455), `i` (4,498), `and` (3,889), `to` (3,767), `of` (3,261) |

---

## 3. Architecture Descriptions

All models share a common structural backbone:
- **Embedding Layer**: $\text{Embedding}(5000, 128, \text{padding\_idx}=0)$
- **Recurrent Hidden Layer**: 128 units
- **Regularization**: Dropout ($p=0.2$) and Gradient Clipping ($\text{norm}=1.0$)
- **Output Layer**: Linear projection to vocabulary size ($5000$)
- **Optimizer**: Adam ($\text{lr}=10^{-3}$)
- **Loss Criterion**: Cross-Entropy Loss

```
Input Tokens: [w1, w2, ..., w8]
       │
       ▼
Embedding Layer (128-dim)
       │
       ▼
Recurrent Core (Simple RNN / BiRNN / LSTM / GRU, 128 units)
       │
       ▼
Final Step Context Representation
       │
       ▼
Dropout (p = 0.2)
       │
       ▼
Dense Projection (Vocab Size = 5000)
       │
       ▼
Softmax Logits -> Predicted Next Word
```

1. **Simple RNN**: Uses standard hyperbolic tangent hidden recurrent updates $h_t = \tanh(W x_t + U h_{t-1} + b)$. Susceptible to vanishing/exploding gradients on long sequences.
2. **Bidirectional RNN**: Processes the sequence in both forward ($1 \to T$) and backward ($T \to 1$) directions. Concatenates forward and backward final hidden vectors ($2 \times 128 = 256$ dimensions) before linear classification.
3. **LSTM**: Implements gating units (Forget, Input, Output gates) and an explicit additive cell state $c_t$ to preserve long-range dependencies and eliminate gradient decay.
4. **GRU**: Streamlines LSTM into two gates (Reset $r_t$ and Update $z_t$), achieving comparable representational power with fewer parameters and lower computational overhead.

---

## 4. Repository Structure

```
.
├── data/
│   ├── dataset_source_link.txt
│   └── tinyshakespeare.txt
├── notebooks/
│   └── text_generation_assignment2.ipynb
├── results/
│   ├── accuracy_comparison.png
│   ├── dataset_stats.json
│   ├── generated_samples.txt
│   ├── loss_curves.png
│   ├── metrics_history.json
│   ├── perplexity_comparison.png
│   ├── quantitative_results.md
│   ├── training_time_comparison.png
│   ├── simple_rnn_best.pt
│   ├── bidirectional_rnn_best.pt
│   ├── lstm_best.pt
│   └── gru_best.pt
├── scripts/
│   ├── build_notebook.py
│   └── generate_submission_docx.py
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── models.py
│   ├── preprocessing.py
│   └── train.py
├── run_experiments.py
├── requirements.txt
├── SakshamGupta_RegNum_FGAI_Assignment2.docx
└── README.md
```

---

## 5. Execution Steps & Dependencies

### Prerequisites
- Python 3.9+
- Recommended: Apple Silicon (MPS), CUDA-enabled GPU, or CPU

### Installation
```bash
# Clone the repository
git clone https://github.com/sakshamgupta/autoregressive-text-generation-fgai.git
cd autoregressive-text-generation-fgai

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run the complete automated training, evaluation, and generation pipeline
python3 run_experiments.py
```

### Interactive Notebook
Open and run all cells in `notebooks/text_generation_assignment2.ipynb`:
```bash
jupyter notebook notebooks/text_generation_assignment2.ipynb
```

---

## 6. Quantitative Results & Evaluation

### Comparison Table
| Architecture | Trainable Parameters | Final Val Loss | Validation Perplexity | Top-1 Accuracy (%) | Total Training Time | Avg Epoch Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 1,318,024 | 5.1671 | 175.40 | 16.34% | 37.9s | 3.79s |
| **Bidirectional RNN** | 1,991,048 | 5.3461 | 209.79 | 15.49% | 42.7s | 4.27s |
| **LSTM** | **1,417,096** | **5.0745** | **159.89** | **16.94%** | **36.0s** | **3.58s** |
| **GRU** | 1,384,072 | 5.1609 | 174.32 | 16.39% | 37.3s | 3.72s |

### Visual Comparisons

#### 1. Training & Validation Loss Curves
![Loss Curves](results/loss_curves.png)

#### 2. Validation Perplexity (Lower is Better)
![Perplexity Comparison](results/perplexity_comparison.png)

#### 3. Top-1 Next-Word Accuracy (%)
![Accuracy Comparison](results/accuracy_comparison.png)

#### 4. Total Training Time (Seconds)
![Training Time Comparison](results/training_time_comparison.png)

---

## 7. Sampling Temperature Analysis

During autoregressive decoding, next-token probabilities are modulated by temperature $T$:
$$P(w_{t} = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluated four regimes: **$T \in \{0.2, 0.5, 1.0, 1.5\}$** across three seed prompts.

### Sample Generations from Seed: *"to be or not to be that is the question"*

| Temperature | Model | Generated Sample | Distinct-1 | Distinct-2 | Qualitative Assessment |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **$T=0.2$** | LSTM | `... of the people, and that i am, and i am, as i will not be a man: i will not be a man, and i will not be so.` | 0.47 | 0.68 | Highly confident but gets trapped in repetitive loop cycles. |
| **$T=0.5$** | **LSTM** | `... and i am, i will not be so can have been a scene of the people, and then. coriolanus: why, if you are incensed not, my lord,` | **0.71** | **0.95** | **Optimal coherence & diversity**. Captures character dialogue dialogue tags and Shakespearean cadence. |
| **$T=1.0$** | LSTM | `... of a heal unlike first wherein you importune speak. menenius: and so issue mark him seen that love me like stones. sicinius: i want not can tell him that help his` | 0.82 | 0.98 | Rich vocabulary, but grammatical binding weakens across clauses. |
| **$T=1.5$** | LSTM | `... is eyes their god's wheel whispering he part london. daughter say me, yielded bear my deed on here henry the bows, lose i do whither far suitors sent at man unknown lieutenant` | 0.89 | 0.98 | High entropy; syntactic fragmentation and loss of contextual sense. |

---

## 8. Conclusion: Best-Performing Model Justification

**LSTM is the most effective model for autoregressive generation on this dataset**:
1. **Lowest Perplexity & Loss**: LSTM achieved a validation perplexity of **159.89** (vs. 174.32 for GRU, 175.40 for Simple RNN, and 209.79 for BiRNN), showing superior probability modeling on unseen tokens.
2. **Highest Accuracy**: Reached the highest next-word top-1 accuracy (**16.94%**).
3. **Execution Efficiency**: Despite having gating units, LSTM achieved the fastest per-epoch training time (**3.58s/epoch**) on Apple Silicon MPS due to optimized matrix kernel fusion.
4. **Inductive Bias for Autoregressive Generation**: Bidirectional RNN struggled comparatively (Perplexity 209.79) because backward sequence processing creates an inference mismatch in strictly left-to-right causal autoregressive generation. In contrast, LSTM's directional memory cells naturally align with autoregressive generation.
5. **Optimal Sampling Setting**: Combining the **LSTM model with $T = 0.5$** provides the ideal trade-off between semantic coherence and lexical diversity.
