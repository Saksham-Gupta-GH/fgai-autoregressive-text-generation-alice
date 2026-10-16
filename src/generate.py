import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter


def sample_with_temperature(logits, temperature=1.0):
    """
    Applies temperature scaling to logits and draws a token from the resulting multinomial distribution.
    - Low T (< 0.5): Greedy-leaning, confident, conservative, prone to repetition.
    - Moderate T (~ 0.7 - 1.0): Balanced creativity and coherence.
    - High T (> 1.2): High diversity, high randomness, higher risk of syntactic breakdown.
    """
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1).item()
    
    # Scale logits
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    
    # Sample from multinomial distribution
    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
    return sampled_idx


def generate_text(model, vocab, seed_text, max_new_tokens=30, seq_length=8, temperature=1.0, device="cpu"):
    """
    Autoregressively generates text token-by-token given a seed sequence.
    Maintains a sliding window of length `seq_length`.
    """
    model.eval()
    from src.preprocessing import clean_and_tokenize
    
    seed_tokens = clean_and_tokenize(seed_text)
    if len(seed_tokens) == 0:
        seed_tokens = ["to", "be"]
        
    generated_tokens = list(seed_tokens)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Take the most recent `seq_length` tokens
            context = generated_tokens[-seq_length:]
            indices = vocab.encode(context)
            
            # If context is shorter than seq_length, pad on the left
            if len(indices) < seq_length:
                indices = [vocab.pad_idx] * (seq_length - len(indices)) + indices
                
            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
            logits = model(input_tensor) # (1, vocab_size)
            logits = logits.squeeze(0)   # (vocab_size)
            
            # Disallow generating <pad> or <unk> directly
            logits[vocab.pad_idx] = -float("inf")
            logits[vocab.unk_idx] = -float("inf")
            
            next_idx = sample_with_temperature(logits, temperature=temperature)
            next_word = vocab.idx_to_word(next_idx)
            generated_tokens.append(next_word)
            
    # Format generated output nicely
    # Reconstruct text respecting punctuation
    formatted_words = []
    for token in generated_tokens:
        if token in {".", ",", "!", "?", ":", ";"} and len(formatted_words) > 0:
            formatted_words[-1] = formatted_words[-1] + token
        else:
            formatted_words.append(token)
            
    return " ".join(formatted_words), generated_tokens


def calculate_generation_metrics(tokens):
    """
    Calculates quantitative metrics on generated text:
    - Distinct-1: Ratio of unique unigrams to total tokens
    - Distinct-2: Ratio of unique bigrams to total bigrams
    - Repetition-Rate: Percentage of duplicate tokens
    """
    if len(tokens) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}
        
    unigrams = tokens
    distinct_1 = len(set(unigrams)) / len(unigrams)
    
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    
    # Count consecutive repetitions (w_i == w_{i+1})
    consecutive_reps = sum(1 for i in range(len(unigrams) - 1) if unigrams[i] == unigrams[i+1])
    consecutive_rate = consecutive_reps / max(1, len(unigrams) - 1)
    
    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "consecutive_repeat_rate": round(consecutive_rate, 4)
    }
