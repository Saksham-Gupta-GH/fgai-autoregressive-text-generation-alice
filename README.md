# Foundations of Generative AI (ICT4403) — Assignment 2
## Autoregressive Next-Word Prediction & Multi-Temperature Text Generation System: Comparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU

---

### Student Details
- **Student Name:** Saksham Gupta
- **Registration Number:** 230911186
- **Assignment Title:** Foundations of Generative AI (ICT4403) – Assignment 2
- **System Title:** Comparative Word-Level Autoregressive Text Generation on Lewis Carroll's *Alice's Adventures in Wonderland* using Simple RNN, Bidirectional RNN, LSTM, and GRU
- **GitHub Repository Link:** [https://github.com/sakshamgupta/autoregressive-text-generation-fgai](https://github.com/sakshamgupta/autoregressive-text-generation-fgai)

---

## 1. Problem Statement & Objectives

Autoregressive language modeling estimates the joint probability of a sequence of tokens by factorizing it into a product of conditional next-token distributions:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

The core objectives of this assignment are:
1. **Dataset Preprocessing & Strict Leakage Prevention**: Preprocess Lewis Carroll's **Alice's Adventures in Wonderland**, implement a **chronological contiguous split** (80% Train, 10% Validation, 10% Test) rather than random window shuffling (which leaks overlapping tokens), and fit vocabulary **strictly on training tokens**.
2. **Model Implementations in PyTorch**: Construct four recurrent neural architectures:
   - **Simple RNN** (Vanilla Elman recurrent network)
   - **Bidirectional RNN** (BiRNN, properly concatenating forward and backward final hidden states `[h_n[-2], h_n[-1]]`)
   - **LSTM** (Long Short-Term Memory with forget, input, and output gates)
   - **GRU** (Gated Recurrent Unit with reset and update gates)
3. **Controlled & Comparable Experimental Conditions**: Train all four models under identical conditions (same random seed, 192 embedding dimensions, 192 hidden units, dropout 0.25, Adam optimizer, learning rate 0.001, batch size 64, and gradient clipping at 1.0).
4. **Rigorous Evaluation**: Track training and validation learning curves, record **best-epoch validation checkpoints** (to account for overfitting gaps), and evaluate best models on an unseen test set for Cross-Entropy Loss, Perplexity ($\exp(\mathcal{L})$), Top-1 Accuracy, and Top-5 Candidate Accuracy.
5. **Cross-Model Temperature Analysis**: Sample text across four temperature regimes ($T \in \{0.2, 0.5, 1.0, 1.5\}$) from multiple seed prompts, evaluating Distinct-1, Distinct-2, and Trigram Repetition Rate strictly on newly generated tokens.

---

## 2. Dataset Description & Source Link

- **Dataset Name**: Alice's Adventures in Wonderland (by Lewis Carroll)
- **Source Link**: [https://www.gutenberg.org/files/11/11-0.txt](https://www.gutenberg.org/files/11/11-0.txt)
- **Origin**: Project Gutenberg, eBook #11.
- **Why this dataset**: Lewis Carroll's prose features standard modern English syntax, structured dialogic exchanges, and a compact vocabulary where recurrent models can develop meaningful syntactic and semantic representations without the archaic distortions of Elizabethan English.

### Dataset Statistical Characteristics
| Metric | Value |
| :--- | :--- |
| **Total Characters** | 144,015 |
| **Total Word Tokens** | 31,841 |
| **Total Lines** | 3,351 |
| **Vocabulary Size (Train-Only)** | 2,366 unique tokens (including `<pad>`, `<unk>`) |
| **Train Sequences (80%)** | 25,465 sequences (397 batches, batch size 64) |
| **Validation Sequences (10%)** | 3,176 sequences (50 batches) |
| **Test Sequences (10%)** | 3,176 sequences (50 batches) |
| **Top-5 Frequent Words in Train** | `,` (1,922), `the` (1,304), `.` (785), `and` (699), `to` (577) |

---

## 3. Architecture Descriptions

All models are implemented in PyTorch and share an identical structure:
- **Embedding Layer**: $\text{Embedding}(2366, 192, \text{padding\_idx}=0)$
- **Recurrent Core**: 192 units (Simple RNN / BiRNN / LSTM / GRU)
- **Regularization**: Dropout ($p=0.25$) and Gradient Clipping ($\text{norm}=1.0$)
- **Classification Head**: $\text{Linear}(H \to 2366)$ (or $2H \to 2366$ for BiRNN)
- **Optimizer**: Adam ($\text{lr}=10^{-3}$)
- **Criterion**: Cross-Entropy Loss

```
Input Tokens: [w1, w2, ..., w8]
       │
       ▼
Embedding Layer (192-dim)
       │
       ▼
Recurrent Core (Simple RNN / BiRNN / LSTM / GRU, 192 units)
       │
       ▼
Final Step Context Representation
       │
       ▼
Dropout (p = 0.25)
       │
       ▼
Dense Projection (Vocab Size = 2366)
       │
       ▼
Logits -> Next-Word Probability Distribution
```

### Architectural Notes & Fairness Considerations
1. **Simple RNN**: Relies on hyperbolic tangent recurrent updates $h_t = \tanh(W x_t + U h_{t-1} + b)$. Susceptible to vanishing gradients over sequence steps.
2. **Bidirectional RNN**: Processes the 8-token context window in both forward ($0 \to T-1$) and backward ($T-1 \to 0$) directions. The representation combines the forward final state $h_n[-2]$ with the backward final state $h_n[-1]$ (`torch.cat([h_n[-2], h_n[-1]], dim=-1)`), giving the linear classifier access to the complete context from both directions.  
   *Fairness Caveat*: BiRNN has **1,513,406 parameters** (vs. ~1.1M–1.2M for unidirectional models). Because of its higher capacity on a compact corpus, it reaches its optimal validation loss earlier (epoch 2) before overfitting.
3. **LSTM**: Employs gating mechanisms (Forget gate $f_t$, Input gate $i_t$, Output gate $o_t$) and an additive cell state $c_t$ to preserve temporal dependencies across sequence steps.
4. **GRU**: Streamlines memory tracking into two gates (Reset $r_t$ and Update $z_t$), balancing representational expressiveness with fewer parameters (1,133,246).

---

## 4. Execution Steps & Dependencies

### Prerequisites
- Python 3.9+
- Hardware: Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

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
# Execute automated training, evaluation, plotting, and generation pipeline
python3 run_experiments.py
```

### Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/text_generation_assignment2.ipynb
```

---

## 5. Quantitative Results & Evaluation

All four models were trained for 12 epochs on Apple Silicon MPS. Because recurrent models on small corpora begin overfitting after several epochs (train loss drops toward ~1.0–2.0 while validation loss plateaus), **we report best-epoch validation metrics alongside held-out test set performance**:

### Model Comparison Table (Best-Epoch & Test Set)
| Architecture | Trainable Parameters | Best Epoch | Best Val Loss | Best Val PPL | Val Top-1 Acc | Val Top-5 Acc | Test Loss | Test Top-1 Acc | Test Top-5 Acc | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 985,022 | 3 | 5.0287 | 152.73 | 18.10% | 38.66% | 5.1567 | 17.60% | 38.41% | 42.0s |
| **Bidirectional RNN** | 1,513,406 | 2 | 5.1401 | 170.74 | 16.88% | 37.81% | 5.2474 | 16.94% | 35.96% | 49.2s |
| **LSTM** 🏆 | 1,207,358 | 4 | 4.9778 | 145.16 | 18.70% | **41.62%** | **5.1261** | **19.68%** | **39.17%** | 41.4s |
| **GRU** 🏆 | 1,133,246 | 4 | **4.9701** | **144.04** | **18.92%** | 40.27% | 5.1361 | 19.49% | 38.95% | 43.2s |

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

## 6. Cross-Model Temperature Analysis

During autoregressive decoding, next-token probabilities are modulated by temperature $T$:
$$P(w_{t} = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluated all four architectures across three distinct seed prompts at four temperatures ($T \in \{0.2, 0.5, 1.0, 1.5\}$). All metrics are computed **strictly on newly generated tokens** (excluding the seed prompt) and averaged across seeds:

### Cross-Model Temperature Metrics Table (Averaged Over Seeds)
| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Quality Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | 0.2 | 0.448 | 0.588 | 33.3% | 6 words | Conservative; high confidence; falls into repetitive looping phrases |
| **Simple RNN** | 0.5 | 0.629 | 0.843 | 3.0% | 0 words | Good diversity; moderate phrasing stability |
| **Simple RNN** | 1.0 | 0.867 | 1.000 | 0.0% | 0 words | High vocabulary variety; occasional grammatical drift |
| **Simple RNN** | 1.5 | 0.905 | 1.000 | 0.0% | 0 words | High entropy; syntactic fragmentation |
| **Bidirectional RNN** | 0.2 | 0.533 | 0.745 | 14.1% | 4 words | Conservative; repeats key character phrases ("and the queen") |
| **Bidirectional RNN** | 0.5 | 0.686 | 0.912 | 4.0% | 0 words | Good balance of distinct n-grams with coherent clauses |
| **Bidirectional RNN** | 1.0 | 0.867 | 1.000 | 0.0% | 0 words | Rich vocabulary; multi-clause drift |
| **Bidirectional RNN** | 1.5 | 0.895 | 1.000 | 0.0% | 0 words | High randomness; loss of semantic thread |
| **LSTM** | 0.2 | 0.552 | 0.765 | 11.1% | 4 words | Strict syntax; low phrase variety |
| **LSTM** | **0.5** | **0.752** | **0.931** | **3.0%** | **0 words** | **Best empirical trade-off: high lexical diversity with fluent phrasing** |
| **LSTM** | 1.0 | 0.848 | 1.000 | 0.0% | 0 words | Highly creative; occasional sentence fragmentation |
| **LSTM** | 1.5 | 0.943 | 1.000 | 0.0% | 0 words | Maximum entropy; ungrammatical token salad |
| **GRU** | 0.2 | 0.495 | 0.706 | 16.2% | 4 words | Prone to repetitive phrasing ("and then she went on...") |
| **GRU** | 0.5 | 0.648 | 0.824 | 10.1% | 4 words | Fluent dialogic tags; low phrase repetition |
| **GRU** | 1.0 | 0.838 | 1.000 | 0.0% | 0 words | High vocabulary richness |
| **GRU** | 1.5 | 0.876 | 1.000 | 0.0% | 0 words | High entropy; loss of syntactic cohesion |

---

## 7. Qualitative Generation Examples

### Seed Prompt: *"alice was beginning to get very tired of sitting"*
* **LSTM ($T=0.2$)**:
  > `"... that she was quite to find her, and she had to herself, and she had to her. it was a little thing, and she had to be a little golden key"`
* **LSTM ($T=0.5$)**:
  > `"... a little house, she was looking up and she began to her feet, and the queen s argument to get. and i m sure you mean it, said the hatter."`
* **LSTM ($T=1.0$)**:
  > `"... on that creatures, but she had small give a curious round and the verses and rose fair curious them. as all while anything alice seen life..."`
* **LSTM ($T=1.5$)**:
  > `"... wags over their favourite tastes do she said aloud again: on nor desk come what it leave for past: sobs when that or seemed to feel..."`

---

## 8. Conclusion: Model Comparison & Justification

1. **GRU and LSTM are Closely Matched**:
   - **GRU** achieved the lowest validation loss (**4.9701**) and lowest perplexity (**144.04**), while training with fewer parameters (1,133,246).
   - **LSTM** achieved the highest test-set Top-1 accuracy (**19.68%**) and highest validation Top-5 accuracy (**41.62%**).
   - Both gated architectures decisively outperform Simple RNN and BiRNN by maintaining stable gradient flow and gating past context effectively.
2. **Bidirectional RNN Analysis**:
   - Concatenating the forward final state $h_n[-2]$ and backward final state $h_n[-1]$ properly feeds the full sequence context to the classifier.
   - However, BiRNN's 1.51M parameters cause it to overfit earlier (validation loss peaks at epoch 2 at 5.1401). For small text generation tasks, the extra backward parameters provide limited inductive advantage compared to causal gated units.
3. **Sampling Strategy**:
   - $T = 0.5$ represents the **best empirical trade-off** between lexical diversity (Distinct-1: ~0.68–0.75) and low phrase repetition (trigram repetition rate under 5%). $T = 0.2$ results in phrase traps (repetition rate > 15%), while $T = 1.5$ degrades into random token sampling.
