import csv
from src.envs.cartpole import CartPole
from src.transformer import heuristic_policy, computeRTG

def collect_episodes(K, num_episodes, max_steps=500):
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)
    episodes = []

    for ep in range(num_episodes):
        states = []
        actions = []
        rewards = []

        state = env.reset()

        for step in range(max_steps):
            states.append(state)

            u = heuristic_policy(state, K)
            state, reward, done = env.step(u)

            actions.append(u)
            rewards.append(reward)

            if done: break

        rtg = computeRTG(rewards)

        episodes.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'rtg': rtg
        })

        print(f"Episode {ep}: {len(states)} steps, RTG={rtg[0]}")
    
    return episodes

def generateData():
    dataset = []
    dataset += collect_episodes(K=8,num_episodes=50)
    dataset += collect_episodes(K=10,num_episodes=70)
    dataset += collect_episodes(K=12,num_episodes=30)

    print(f"Total Episodes: {len(dataset)}")

    import pickle
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print("data saved to data/dataset.pkl")

def create_sequences(episodes, seq_len=20):
    sequences = []

    for episode in episodes:
        states = episode['states']
        actions = episode['actions']
        rtg = episode['rtg']
        rewards = episode['rewards']

        episode_len = len(states)
        for i in range(0, episode_len, seq_len):
            chunk_s = states[i:(i+seq_len)]
            chunk_a = actions[i:(i+seq_len)]
            chunk_r = rtg[i:(i+seq_len)]
            chunk_rewards = rewards[i:i+seq_len]
            
            length = len(chunk_s)
            if length < seq_len:
                remaining = seq_len - length
                chunk_s += [[0.0, 0.0, 0.0, 0.0],] * remaining
                chunk_a += [0.0] * remaining
                chunk_r += [0.0] *remaining
                chunk_rewards = chunk_rewards + [0.0] * remaining

                mask = [1]*length + [0]*remaining

            else:
                mask = [1]*seq_len

            sequences.append({
                'states': chunk_s,
                'actions': chunk_a,
                'rtg': chunk_r,
                'rewards': chunk_rewards,
                'mask': mask
            })

    return sequences

def train_val_split(sequences, val_ratio=0.2):
    """
    Input:
        sequences
        validation ratio -> default -> 20%
    Output:
        train_sequences, val_sequences
    """
    import random

    shuffled = sequences.copy()
    random.shuffle(shuffled)

    val_size = int(len(sequences)*val_ratio)

    val_sequences, train_sequences = shuffled[:val_size], shuffled[val_size:]
    return train_sequences, val_sequences

def save_dataset(train_sequences, val_sequences, filepath="data/dataset.pt"):
    """
    Save dataset to PyTorch format with statistics
    
    Args:
        train_sequences: list of training sequences
        val_sequences: list of validation sequences
        filepath: where to save
    """
    import torch
    import numpy as np
    
    # Compute statistics on training data
    all_states = []
    all_rtgs = []
    
    for seq in train_sequences:
        # Only use real data (where mask=1)
        for i, mask_val in enumerate(seq['mask']):
            if mask_val == 1:
                all_states.append(seq['states'][i])
                all_rtgs.append(seq['rtg'][i])
    
    states_array = np.array(all_states)  # Shape: (N, 4)
    rtgs_array = np.array(all_rtgs)      # Shape: (N,)
    
    # Calculate stats
    state_mean = states_array.mean(axis=0).tolist()
    state_std = states_array.std(axis=0).tolist()
    rtg_mean = float(rtgs_array.mean())
    rtg_std = float(rtgs_array.std())
    rtg_max = float(rtgs_array.max())
    
    # Create dataset dict
    dataset = {
        'train': train_sequences,
        'val': val_sequences,
        'stats': {
            'state_mean': state_mean,
            'state_std': state_std,
            'rtg_mean': rtg_mean,
            'rtg_std': rtg_std,
            'rtg_max': rtg_max,
            'n_train': len(train_sequences),
            'n_val': len(val_sequences)
        }
    }
    
    # Save
    torch.save(dataset, filepath)
    
    # Print stats
    print(f"✓ Saved dataset to {filepath}")
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")
    print(f"  State mean: {state_mean}")
    print(f"  State std: {state_std}")
    print(f"  RTG range: [{rtgs_array.min():.1f}, {rtg_max:.1f}]")
    print(f"  RTG mean±std: {rtg_mean:.1f}±{rtg_std:.1f}")


def make_seq_and_save():
    import pickle
    # Load episodes
    print("Loading episodes...")
    with open("data/dataset.pkl", "rb") as f:
        episodes = pickle.load(f)
    print(f"✓ Loaded {len(episodes)} episodes")
    
    # Create sequences
    print("\nCreating sequences...")
    sequences = create_sequences(episodes, seq_len=20)
    print(f"✓ Created {len(sequences)} sequences")
    
    # Train/val split
    print("\nSplitting train/val...")
    train_seqs, val_seqs = train_val_split(sequences, val_ratio=0.2)
    
    # Save
    print("\nSaving dataset...")
    save_dataset(train_seqs, val_seqs, filepath="data/dataset.pt")