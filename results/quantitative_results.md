# Model Comparison Table (Best-Epoch and Test Set)

| Architecture | Parameters | Best Epoch | Best Val Loss | Best Val PPL | Val Top-1 | Val Top-5 | Test Loss | Test Top-1 | Test Top-5 | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 798,484 | 4 | 4.0741 | 58.80 | 19.73% | 44.97% | 4.1584 | 20.80% | 45.28% | 26.1s |
| **Bidirectional RNN** | 1,262,868 | 2 | 4.2804 | 72.27 | 18.66% | 42.20% | 4.3192 | 18.47% | 42.26% | 27.7s |
| **LSTM** | 1,193,236 | 5 | 3.9903 | 54.07 | 21.87% | 46.07% | 4.0576 | 23.29% | 47.23% | 23.0s |
| **GRU** | 1,061,652 | 3 | 4.0353 | 56.56 | 20.55% | 45.72% | 4.0900 | 22.53% | 46.51% | 25.8s |


# Cross-Model Temperature Analysis Table (Averaged Over Seeds)

| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Behaviour |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | 0.2 | 0.571 | 0.716 | 0.202 | 0 words | Conservative, high confidence, prone to phrase repetition |
| **Simple RNN** | 0.5 | 0.600 | 0.833 | 0.061 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **Simple RNN** | 1.0 | 0.867 | 0.990 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **Simple RNN** | 1.5 | 0.914 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **Bidirectional RNN** | 0.2 | 0.581 | 0.804 | 0.121 | 6 words | Conservative, high confidence, prone to phrase repetition |
| **Bidirectional RNN** | 0.5 | 0.714 | 0.931 | 0.010 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **Bidirectional RNN** | 1.0 | 0.867 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **Bidirectional RNN** | 1.5 | 0.943 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **LSTM** | 0.2 | 0.571 | 0.765 | 0.121 | 0 words | Conservative, high confidence, prone to phrase repetition |
| **LSTM** | 0.5 | 0.667 | 0.902 | 0.050 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **LSTM** | 1.0 | 0.857 | 1.000 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **LSTM** | 1.5 | 0.895 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
| **GRU** | 0.2 | 0.343 | 0.412 | 0.525 | 6 words | Conservative, high confidence, prone to phrase repetition |
| **GRU** | 0.5 | 0.733 | 0.931 | 0.000 | 0 words | Optimal trade-off: high lexical diversity with fluent phrasing |
| **GRU** | 1.0 | 0.848 | 0.980 | 0.000 | 0 words | High vocabulary variety; occasional grammatical drift |
| **GRU** | 1.5 | 0.914 | 1.000 | 0.000 | 0 words | High entropy, syntactic fragmentation, loss of coherence |
