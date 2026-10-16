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
| **Simple RNN** | 0.2 | 0.571 | 0.716 | 0.202 | 0 words | Repetitive phrasing loops; repeats 'cauldron of the door' motif |
| **Simple RNN** | 0.5 | 0.600 | 0.833 | 0.061 | 0 words | Moderate fluency with local repetition ('door that makes the world would not') |
| **Simple RNN** | 1.0 | 0.867 | 0.990 | 0.000 | 0 words | High lexical variety; syntactic drift across longer clauses |
| **Simple RNN** | 1.5 | 0.914 | 1.000 | 0.000 | 0 words | Excessive entropy; severe syntactic breakdown and ungrammatical token salad |
| **Bidirectional RNN** | 0.2 | 0.581 | 0.804 | 0.121 | 6 words | Severe looping traps; enters 6-word cycle 'and the moral of course' |
| **Bidirectional RNN** | 0.5 | 0.714 | 0.931 | 0.010 | 0 words | Lively dialogue phrasing ('cried the pigeon, i don t think you know'); zero loops |
| **Bidirectional RNN** | 1.0 | 0.867 | 1.000 | 0.000 | 0 words | Rich character vocabulary; clauses fragment over multi-word spans |
| **Bidirectional RNN** | 1.5 | 0.943 | 1.000 | 0.000 | 0 words | High disorder; random lexicon emission without coherent syntactic bindings |
| **LSTM** | 0.2 | 0.571 | 0.765 | 0.121 | 0 words | Strict syntax; repeats dominant character motifs ('queen s argument was') |
| **LSTM** | 0.5 | 0.667 | 0.902 | 0.050 | 0 words | Optimal narrative coherence; fluent story prose ('the march hare said in a hurry') |
| **LSTM** | 1.0 | 0.857 | 1.000 | 0.000 | 0 words | Imaginative lexical combinations ('way of soup'); slight clause drift |
| **LSTM** | 1.5 | 0.895 | 1.000 | 0.000 | 0 words | Entropy collapse; erratic word sequencing ('face high before without more pig') |
| **GRU** | 0.2 | 0.343 | 0.412 | 0.525 | 6 words | Extreme phrase trapping (52.5% repetition); loops on 'and she tried to herself' |
| **GRU** | 0.5 | 0.733 | 0.931 | 0.000 | 0 words | Clean conversational fluency with zero repetition loops ('said the cat... went on') |
| **GRU** | 1.0 | 0.848 | 0.980 | 0.000 | 0 words | Diverse lexicon; minor grammatical degradation in dialogue tags |
| **GRU** | 1.5 | 0.914 | 1.000 | 0.000 | 0 words | Unconstrained random sampling; complete loss of semantic and syntactic thread |
