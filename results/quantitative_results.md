# Model Comparison Table

| Architecture | Parameters | Final Val Loss | Perplexity | Top-1 Accuracy | Top-5 Accuracy | Total Time (s) | Avg Epoch Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 724,097 | 4.9572 | 142.20 | 20.28% | 40.72% | 54.1s | 4.51s |
| **Bidirectional RNN** | 1,101,313 | 5.1979 | 180.89 | 19.37% | 40.10% | 59.1s | 4.92s |
| **LSTM** | 823,169 | 4.8746 | 130.92 | 20.22% | 42.02% | 48.1s | 4.00s |
| **GRU** | 790,145 | 4.9834 | 145.98 | 20.09% | 40.81% | 50.4s | 4.19s |
