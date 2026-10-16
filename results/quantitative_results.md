# Model Comparison Table (Best-Epoch and Test Set)

| Architecture | Parameters | Best Epoch | Best Val Loss | Best Val PPL | Val Top-1 | Val Top-5 | Test Loss | Test Top-1 | Test Top-5 | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 985,022 | 3 | 5.0287 | 152.73 | 18.10% | 38.66% | 5.1567 | 17.60% | 38.41% | 42.0s |
| **Bidirectional RNN** | 1,513,406 | 2 | 5.1401 | 170.74 | 16.88% | 37.81% | 5.2474 | 16.94% | 35.96% | 49.2s |
| **LSTM** | 1,207,358 | 4 | 4.9778 | 145.16 | 18.70% | 41.62% | 5.1261 | 19.68% | 39.17% | 41.4s |
| **GRU** | 1,133,246 | 4 | 4.9701 | 144.04 | 18.92% | 40.27% | 5.1361 | 19.49% | 38.95% | 43.2s |


# Cross-Model Temperature Analysis Table (Averaged Over Seeds)

| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Behaviour |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | 0.2 | 0.448 | 0.588 | 0.333 | 6 words | Conservative, high confidence, prone to phrase repetition |
| **Simple RNN** | 0.5 | 0.629 | 0.843 | 0.030 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **Simple RNN** | 1.0 | 0.867 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **Simple RNN** | 1.5 | 0.905 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **Bidirectional RNN** | 0.2 | 0.533 | 0.745 | 0.141 | 4 words | Conservative, high confidence, prone to phrase repetition |
| **Bidirectional RNN** | 0.5 | 0.686 | 0.912 | 0.040 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **Bidirectional RNN** | 1.0 | 0.867 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **Bidirectional RNN** | 1.5 | 0.895 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **LSTM** | 0.2 | 0.552 | 0.765 | 0.111 | 4 words | Conservative, high confidence, prone to phrase repetition |
| **LSTM** | 0.5 | 0.752 | 0.931 | 0.030 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **LSTM** | 1.0 | 0.848 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **LSTM** | 1.5 | 0.943 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **GRU** | 0.2 | 0.495 | 0.706 | 0.162 | 4 words | Conservative, high confidence, prone to phrase repetition |
| **GRU** | 0.5 | 0.648 | 0.824 | 0.101 | 4 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **GRU** | 1.0 | 0.838 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **GRU** | 1.5 | 0.876 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
