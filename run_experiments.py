import os
import json
import torch
import numpy as np

from src.preprocessing import prepare_data
from src.models import get_model
from src.train import train_model
from src.evaluate import plot_training_results, format_comparison_table
from src.generate import generate_text, calculate_generation_metrics


def run_all(
    data_path="data/tinyshakespeare.txt",
    results_dir="results",
    epochs=10,
    seq_length=8,
    embedding_dim=128,
    hidden_dim=128,
    batch_size=128,
    max_vocab_size=5000,
    max_tokens=60000,  # Fast training subset for rapid, rigorous experimentation
    seed=42
):
    # Set seeds for strict reproducibility across all models
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    os.makedirs(results_dir, exist_ok=True)
    
    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using Compute Device: {device.upper()}")
    
    # 1. Dataset Preprocessing & Characterization
    print("\n[Step 1/4] Preprocessing Dataset and Constructing Autoregressive Sequences...")
    data_bundle = prepare_data(
        file_path=data_path,
        seq_length=seq_length,
        max_vocab_size=max_vocab_size,
        batch_size=batch_size,
        max_samples=max_tokens,
        seed=seed
    )
    
    vocab = data_bundle["vocab"]
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    stats = data_bundle["stats"]
    
    print("\n--- Dataset Characteristics ---")
    print(f"Total Raw Characters : {stats['total_characters']:,}")
    print(f"Total Raw Tokens     : {stats['total_tokens']:,}")
    print(f"Total Lines          : {stats['total_lines']:,}")
    print(f"Vocabulary Size      : {len(vocab):,} unique words")
    print(f"Training Batches     : {len(train_loader)} (Batch Size: {batch_size})")
    print(f"Validation Batches   : {len(val_loader)}")
    
    # Save dataset stats
    with open(os.path.join(results_dir, "dataset_stats.json"), "w") as f:
        # Convert non-serializable tuples
        stats_serializable = dict(stats)
        stats_serializable["top_10_words"] = [list(item) for item in stats["top_10_words"]]
        json.dump(stats_serializable, f, indent=4)
        
    # 2. Model Training under Comparable Conditions
    print("\n[Step 2/4] Training All 4 Models Under Identical Experimental Conditions...")
    model_types = ["Simple RNN", "Bidirectional RNN", "LSTM", "GRU"]
    trained_models = {}
    histories = {}
    
    for m_name in model_types:
        # Re-seed before each model instantiation to ensure identical weight initialization seeds
        torch.manual_seed(seed)
        model = get_model(m_name, vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=0.2)
        ckpt_path = os.path.join(results_dir, f"{m_name.lower().replace(' ', '_')}_best.pt")
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=0.001,
            device=device,
            checkpoint_path=ckpt_path
        )
        
        # Load best checkpoint weights
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        trained_models[m_name] = model
        histories[m_name] = history

    # 3. Quantitative Evaluation & Plot Generation
    print("\n[Step 3/4] Generating Comparative Visualizations & Evaluation Table...")
    plot_training_results(histories, save_dir=results_dir)
    table_md = format_comparison_table(histories)
    print("\n" + table_md)
    
    with open(os.path.join(results_dir, "quantitative_results.md"), "w") as f:
        f.write("# Model Comparison Table\n\n" + table_md)
        
    with open(os.path.join(results_dir, "metrics_history.json"), "w") as f:
        json.dump(histories, f, indent=4)

    # 4. Temperature-Based Text Generation & Qualitative Analysis
    print("\n[Step 4/4] Generating Text Sequences Across Temperatures & Analyzing Coherence...")
    seed_prompts = [
        "to be or not to be that is the question",
        "all the world is a stage and all the men",
        "what light through yonder window breaks it is"
    ]
    temperatures = [0.2, 0.5, 1.0, 1.5]
    
    samples_report = []
    samples_report.append("=" * 80)
    samples_report.append("AUTOREGRESSIVE TEXT GENERATION - TEMPERATURE INFLUENCE ANALYSIS")
    samples_report.append("=" * 80 + "\n")
    
    generation_summary = {}

    for seed_text in seed_prompts:
        samples_report.append(f"\n{'#' * 70}")
        samples_report.append(f"SEED PROMPT: \"{seed_text}\"")
        samples_report.append(f"{'#' * 70}\n")
        
        for m_name in model_types:
            model = trained_models[m_name]
            samples_report.append(f"\n--- Model: {m_name} ---")
            
            for temp in temperatures:
                gen_text, gen_tokens = generate_text(
                    model=model,
                    vocab=vocab,
                    seed_text=seed_text,
                    max_new_tokens=35,
                    seq_length=seq_length,
                    temperature=temp,
                    device=device
                )
                metrics = calculate_generation_metrics(gen_tokens)
                
                entry = (
                    f"[T = {temp:3.1f}] -> {gen_text}\n"
                    f"          (Distinct-1: {metrics['distinct_1']:.2f}, "
                    f"Distinct-2: {metrics['distinct_2']:.2f}, "
                    f"Consecutive Repetition: {metrics['consecutive_repeat_rate']:.2f})\n"
                )
                samples_report.append(entry)
                
                key = f"{m_name}_T{temp}"
                if key not in generation_summary:
                    generation_summary[key] = []
                generation_summary[key].append(metrics)
                
    with open(os.path.join(results_dir, "generated_samples.txt"), "w") as f:
        f.write("\n".join(samples_report))
        
    print(f"\nAll experiments complete! Results saved to '{results_dir}/'")
    return histories, trained_models, vocab


if __name__ == "__main__":
    run_all()
