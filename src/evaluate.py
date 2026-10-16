import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(histories, save_dir="results"):
    """
    Plots and saves comparative learning curves and performance bar charts:
    1. Training & Validation Loss
    2. Validation Perplexity
    3. Validation Accuracy
    4. Training Time per Model
    """
    os.makedirs(save_dir, exist_ok=True)
    models = list(histories.keys())
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    # 1. Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, model_name in enumerate(models):
        hist = histories[model_name]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2)
        axes[1].plot(epochs, hist["val_loss"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2)
        
    axes[0].set_title("Training Loss across Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Validation Loss across Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    
    # 2. Perplexity Curves
    plt.figure(figsize=(8, 5))
    for i, model_name in enumerate(models):
        hist = histories[model_name]
        epochs = range(1, len(hist["val_perplexity"]) + 1)
        plt.plot(epochs, hist["val_perplexity"], label=f"{model_name}", color=colors[i % len(colors)], linewidth=2, marker="o", markersize=4)
        
    plt.title("Validation Perplexity Comparison (Lower is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Perplexity (exp(Val Loss))", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    ppl_path = os.path.join(save_dir, "perplexity_comparison.png")
    plt.savefig(ppl_path, dpi=300)
    plt.close()
    
    # 3. Accuracy Bar Chart (Top-1 and Top-5 Accuracy)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.35
    
    top1_accs = [histories[m]["final_val_acc"] * 100 for m in models]
    top5_accs = [histories[m].get("final_val_top5_acc", histories[m]["final_val_acc"]) * 100 for m in models]
    
    bars1 = plt.bar(x - width/2, top1_accs, width, label="Top-1 Accuracy", color="#2b5c8f")
    bars2 = plt.bar(x + width/2, top5_accs, width, label="Top-5 Accuracy", color="#1b9e77")
    
    plt.title("Next-Word Validation Accuracy: Top-1 vs. Top-5 Candidates (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, max(top5_accs) * 1.25)
    
    for bar in bars1:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    
    # 4. Training Time Bar Chart
    plt.figure(figsize=(8, 5))
    total_times = [histories[m]["total_time"] for m in models]
    bars = plt.bar(models, total_times, color=colors[:len(models)], width=0.5)
    plt.title("Total Training Duration (Seconds)", fontsize=13, fontweight="bold")
    plt.ylabel("Seconds", fontsize=11)
    plt.ylim(0, max(total_times) * 1.25)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}s",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    time_path = os.path.join(save_dir, "training_time_comparison.png")
    plt.savefig(time_path, dpi=300)
    plt.close()
    
    print(f"Saved plots to {save_dir}/")


def format_comparison_table(histories):
    """Generates markdown comparison table."""
    header = "| Architecture | Parameters | Final Val Loss | Perplexity | Top-1 Accuracy | Top-5 Accuracy | Total Time (s) | Avg Epoch Time (s) |\n"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    rows = ""
    for name, hist in histories.items():
        top5 = hist.get("final_val_top5_acc", hist["final_val_acc"])
        rows += (f"| **{name}** | {hist['param_count']:,} | {hist['final_val_loss']:.4f} | "
                 f"{hist['final_perplexity']:.2f} | {hist['final_val_acc']*100:.2f}% | "
                 f"{top5*100:.2f}% | {hist['total_time']:.1f}s | {hist['avg_epoch_time']:.2f}s |\n")
    return header + divider + rows
