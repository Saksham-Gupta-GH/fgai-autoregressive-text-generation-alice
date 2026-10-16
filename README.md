# Foundations of Generative AI (ICT4403) — Assignment 2
## Autoregressive Text Generation System: Comparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU

---

## 1. Problem Statement & Objectives

Autoregressive language models predict each succeeding token conditioned on previously generated tokens:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

The objectives of this assignment are:
1. **Dataset Selection & Preparation**: Preprocess Lewis Carroll's **Alice's Adventures in Wonderland**, construct a clean word-level vocabulary, and create sliding-window input–target sequences $([w_{t-k}, \dots, w_{t-1}] \to w_t)$.
2. **Model Implementations**: Build four recurrent neural architectures from scratch:
   - **Simple RNN** (Vanilla Elman RNN)
   - **Bidirectional RNN** (BiRNN)
   - **LSTM** (Long Short-Term Memory)
   - **GRU** (Gated Recurrent Unit)
3. **Controlled Comparison**: Train all four models under **identical, comparable experimental conditions** (same seed, learning rate, loss function, optimizer, batch size, embedding dimension, and hidden units).
4. **Quantitative & Qualitative Evaluation**: Compare models using:
   - Quantitative: Cross-Entropy Loss, Perplexity ($\exp(\mathcal{L})$), Top-1 Accuracy, Top-5 Candidate Accuracy, and Training Duration.
   - Qualitative: Syntactic coherence, contextual consistency, Distinct-1 / Distinct-2 diversity, and n-gram repetition.
5. **Temperature Analysis**: Autoregressively generate text from seed prompts across four sampling temperatures ($T = 0.2, 0.5, 1.0, 1.5$) to evaluate the trade-off between randomness, repetition, and fluency.

---

## 2. Dataset Description & Source Link

- **Dataset Name**: Alice's Adventures in Wonderland (by Lewis Carroll)
- **Source Link**: [https://www.gutenberg.org/files/11/11-0.txt](https://www.gutenberg.org/files/11/11-0.txt)
- **Origin**: Project Gutenberg, eBook #11.
- **Why this dataset**: Unlike archaic Elizabethan English (e.g. Shakespeare), *Alice in Wonderland* provides clean, modern syntactic structures, consistent conversational prose, and a compact vocabulary where recurrent models can develop strong grammatical and semantic coherence.

### Dataset Statistical Characteristics
| Metric | Value |
| :--- | :--- |
| **Total Characters** | 144,015 |
| **Total Word Tokens** | 31,841 |
| **Total Lines** | 3,351 |
| **Vocabulary Size** | 2,689 unique words (including `<pad>`, `<unk>`) |
| **Top-5 Frequent Words** | `,` (2,427), `the` (1,643), `.` (986), `and` (871), `to` (729) |

---

## 3. Architecture Descriptions

All models share a common structural backbone:
- **Embedding Layer**: $\text{Embedding}(2689, 128, \text{padding\_idx}=0)$
- **Recurrent Hidden Layer**: 128 units
- **Regularization**: Dropout ($p=0.2$) and Gradient Clipping ($\text{norm}=1.0$)
- **Output Layer**: Linear projection to vocabulary size ($2689$)
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
Dense Projection (Vocab Size = 2689)
       │
       ▼
Softmax Logits -> Predicted Next Word
```

1. **Simple RNN**: Uses standard hyperbolic tangent hidden recurrent updates $h_t = \tanh(W x_t + U h_{t-1} + b)$.
2. **Bidirectional RNN**: Encodes sequences in forward ($1 \to T$) and backward ($T \to 1$) directions, concatenating hidden representations before output projection ($2 \times 128 = 256$ dimensions).
3. **LSTM**: Employs gating units (Forget, Input, Output gates) and an explicit cell state $c_t$ to counter vanishing gradients and track long-term narrative dependencies.
4. **GRU**: Uses update and reset gates to balance memory retention with computational efficiency.

---

## 4. Repository Structure

```
.
├── data/
│   ├── alice_in_wonderland.txt
│   └── dataset_source_link.txt
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
- Recommended: Apple Silicon (MPS), CUDA GPU, or CPU

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
| Architecture | Trainable Parameters | Final Val Loss | Validation Perplexity | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Total Training Time | Avg Epoch Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 724,097 | 4.9572 | 142.20 | 20.28% | 40.72% | 54.1s | 4.51s |
| **Bidirectional RNN** | 1,101,313 | 5.1979 | 180.89 | 19.37% | 40.10% | 59.1s | 4.92s |
| **LSTM** 🏆 | **823,169** | **4.8746** | **130.92** | **20.22%** | **42.02%** | **48.1s** | **4.00s** |
| **GRU** | 790,145 | 4.9834 | 145.98 | 20.09% | 40.81% | 50.4s | 4.19s |

### Visual Comparisons

#### 1. Training & Validation Loss Curves
![Loss Curves](results/loss_curves.png)

#### 2. Validation Perplexity (Lower is Better)
![Perplexity Comparison](results/perplexity_comparison.png)

#### 3. Top-1 vs. Top-5 Next-Word Accuracy (%)
![Accuracy Comparison](results/accuracy_comparison.png)

#### 4. Total Training Time (Seconds)
![Training Time Comparison](results/training_time_comparison.png)

---

## 7. Sampling Temperature Analysis

During autoregressive decoding, next-token probabilities are scaled by temperature $T$:
$$P(w_{t} = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluated four regimes: **$T \in \{0.2, 0.5, 1.0, 1.5\}$** across three seed prompts from the novel.

### Sample Generations from Seed: *"alice was beginning to get very tired of sitting"*

| Temperature | Model | Generated Sample | Distinct-1 | Distinct-2 | Qualitative Assessment |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **$T=0.2$** | LSTM | `... that she was quite to find her, and she had to herself, and she had to her. it was a little thing, and she had to be a little golden key` | 0.59 | 0.79 | Highly confident and grammatically sound, but exhibits cyclical repetition. |
| **$T=0.5$** | **LSTM** | `... a little house, she was looking up and she began to her feet, and the queen s argument to get. and i m sure you mean it, said the hatter.` | **0.75** | **0.98** | **Optimal coherence & diversity**. Features authentic characters (Hatter, Queen) and natural English dialogue. |
| **$T=1.0$** | LSTM | `... on that creatures, but she had small give a curious round and the verses and rose fair curious them. as all while anything alice seen life; and he now and mine since` | 0.89 | 1.00 | Rich vocabulary, but multi-clause semantic binding begins to wander. |
| **$T=1.5$** | LSTM | `... wags over their favourite tastes do she said aloud again: on nor desk come what it leave for past: sobs when that or seemed to feel it s get well? she follows` | 0.89 | 1.00 | High entropy; syntactic fragmentation and loss of contextual sense. |

---

## 8. Conclusion: Best-Performing Model Justification

**LSTM is the most effective architecture for autoregressive generation on this dataset**:
1. **Lowest Perplexity & Loss**: Achieved the best validation perplexity of **130.92** and lowest validation loss of **4.8746**.
2. **Highest Candidate Accuracy**: Reached the highest Top-5 candidate accuracy (**42.02%**) and competitive Top-1 accuracy (**20.22%**).
3. **Execution Efficiency**: Trained in **48.1s total** (4.00s/epoch) on Apple Silicon MPS.
4. **Why Bidirectional RNN Lags Behind**: BiRNN yielded higher perplexity (180.89) because processing the sequence in reverse during training creates an inductive discrepancy when the model is tested in causal, left-to-right generation.
5. **Optimal Sampling Setting**: Combining **LSTM with $T = 0.5$** delivers the highest quality autoregressive generation, avoiding both repetitive loops ($T=0.2$) and chaotic word salad ($T=1.5$).
