import time
import math
import torch
import torch.nn as nn
import numpy as np


def calculate_accuracy(logits, targets):
    """Computes Top-1 classification accuracy."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in vanilla/BiRNN
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct_preds += (preds == targets).sum().item()
        total_preds += targets.size(0)
        
    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_preds += (preds == targets).sum().item()
            total_preds += targets.size(0)
            
    val_loss = running_loss / total_preds
    val_acc = correct_preds / total_preds
    val_perplexity = math.exp(min(val_loss, 20))  # numerical guard against overflow
    return val_loss, val_acc, val_perplexity


def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device="cpu", checkpoint_path=None):
    """
    Trains a model under controlled, comparable conditions.
    Tracks loss, perplexity, top-1 accuracy, and wall-clock training time.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_perplexity": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    total_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training: {model.name} on {device.upper()}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_perplexity"].append(val_ppl)
        history["epoch_times"].append(epoch_duration)
        
        if val_loss < best_val_loss and checkpoint_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            
        print(f"Epoch {epoch:2d}/{epochs:2d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:7.2f} | "
              f"Val Acc: {val_acc*100:5.2f}% | "
              f"Time: {epoch_duration:.2f}s")
        
    total_time = time.time() - total_start_time
    history["total_time"] = total_time
    history["avg_epoch_time"] = np.mean(history["epoch_times"])
    history["best_val_loss"] = best_val_loss
    history["final_val_loss"] = history["val_loss"][-1]
    history["final_val_acc"] = history["val_acc"][-1]
    history["final_perplexity"] = history["val_perplexity"][-1]
    history["param_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{model.name} completed in {total_time:.1f}s (Avg: {history['avg_epoch_time']:.2f}s/epoch).")
    return history
