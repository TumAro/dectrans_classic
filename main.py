from experiments.zero_force import zeroForce
from experiments.rollout_random import randomForce
from experiments.heuristic import heuristicForce
from experiments.inspect_data import inspect_data
from src.dataset import make_seq_and_save
import torch
import os

def train():
    """Train the Decision Transformer"""
    from torch.utils.data import Dataset, DataLoader
    from src.transformer import Pipeline
    from src.train_dt import DTDataset, train

    print("\n" + "="*50)
    print("TRAINING DECISION TRANSFORMER")
    print("="*50)

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
        dimension=(4, 1, 1),
        head_count=4,
        layer_count=2
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
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

def evaluate():
    """Evaluate the trained model"""
    from src.transformer import Pipeline
    from src.eval_policy import eval_policy

    print("\n" + "="*50)
    print("EVALUATING POLICY")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Pipeline(
        dimension=(4, 1, 1),
        head_count=4,
        layer_count=2
    ).to(device)

    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from step {checkpoint['step']} (val_loss={checkpoint['val_loss']:.4f})")

    # Evaluate
    results = eval_policy(
        model,
        target_rtg=200.0,
        context_len=20,
        num_episodes=20,
        device=device,
        render_best=True
    )

if __name__ == "__main__":
    # make_seq_and_save()  # Already done - uncomment if you need to regenerate data
    # inspect_data()

    # train()  # Already trained
    evaluate()
