import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.transformer import Pipeline
import numpy as np

class DTDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'states': torch.tensor(seq['states'], dtype=torch.float32),
            'actions': torch.tensor(seq['actions'], dtype=torch.float32).unsqueeze(-1),
            'rtg': torch.tensor(seq['rtg'], dtype=torch.float32).unsqueeze(-1),
            'mask': torch.tensor(seq['mask'], dtype=torch.float32)
        }

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        rtg = batch['rtg'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        predicted_actions = model(states, actions, rtg)

        # Loss: predict next action (teacher forcing)
        # predicted_actions: (batch, seq_len, 1)
        # actions: (batch, seq_len, 1)
        loss = ((predicted_actions - actions) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()

    return total_loss / total_samples

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            mask = batch['mask'].to(device)

            predicted_actions = model(states, actions, rtg)
            loss = ((predicted_actions - actions) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()

            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

    return total_loss / total_samples

def get_lr_schedule(step, warmup_steps=500, max_steps=10000):
    """Linear warmup + cosine decay"""
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

def train(
    model,
    train_loader,
    val_loader,
    num_steps=10000,
    lr=1e-3,
    device='cpu',
    checkpoint_path='checkpoints/best_model.pt',
    log_every=100
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_loss = float('inf')
    step = 0

    print(f"Training for {num_steps} steps...")
    print(f"Device: {device}")

    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break

            # Update learning rate
            lr_mult = get_lr_schedule(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_mult

            # Train step
            model.train()
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            mask = batch['mask'].to(device)

            predicted_actions = model(states, actions, rtg)
            loss = ((predicted_actions - actions) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log
            if step % log_every == 0:
                val_loss = eval_epoch(model, val_loader, device)
                print(f"Step {step}/{num_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr * lr_mult:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")

            step += 1

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return best_val_loss

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = torch.load('data/dataset.pt')
    train_sequences = dataset['train']
    val_sequences = dataset['val']
    print(f"Train: {len(train_sequences)} sequences")
    print(f"Val: {len(val_sequences)} sequences")

    # Create dataloaders
    train_dataset = DTDataset(train_sequences)
    val_dataset = DTDataset(val_sequences)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pipeline(
        dimension=(4, 1, 1),  # state_dim=4, action_dim=1, rtg_dim=1
        head_count=4,
        layer_count=2
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    import os
    os.makedirs('checkpoints', exist_ok=True)

    train(
        model,
        train_loader,
        val_loader,
        num_steps=10000,
        lr=1e-3,
        device=device,
        checkpoint_path='checkpoints/best_model.pt'
    )
