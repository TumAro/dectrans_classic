import torch
import numpy as np
from src.envs.cartpole import CartPole
from src.transformer import Pipeline
import matplotlib.pyplot as plt

def eval_policy(
    model,
    target_rtg=200.0,
    context_len=20,
    num_episodes=20,
    max_steps=500,
    device='cpu',
    render_best=False
):
    """
    Evaluate the Decision Transformer policy

    Args:
        model: trained Pipeline model
        target_rtg: desired return-to-go to condition on
        context_len: how many past timesteps to use
        num_episodes: number of eval episodes
        max_steps: max steps per episode
        device: cpu or cuda
        render_best: if True, save trajectory of best episode
    """
    model.eval()
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)

    episode_returns = []
    all_trajectories = []

    for ep in range(num_episodes):
        state = env.reset()

        # History buffers (will grow up to context_len)
        states_history = [state]
        actions_history = [0.0]  # dummy first action
        rtg_history = [target_rtg]

        episode_reward = 0
        trajectory = []

        for step in range(max_steps):
            # Prepare context (last context_len timesteps)
            ctx_states = states_history[-context_len:]
            ctx_actions = actions_history[-context_len:]
            ctx_rtg = rtg_history[-context_len:]

            # Pad if needed
            if len(ctx_states) < context_len:
                pad_len = context_len - len(ctx_states)
                ctx_states = [[0.0, 0.0, 0.0, 0.0]] * pad_len + ctx_states
                ctx_actions = [0.0] * pad_len + ctx_actions
                ctx_rtg = [0.0] * pad_len + ctx_rtg

            # Convert to tensors
            states_t = torch.tensor(ctx_states, dtype=torch.float32).unsqueeze(0).to(device)
            actions_t = torch.tensor(ctx_actions, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            rtg_t = torch.tensor(ctx_rtg, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            # Predict action
            with torch.no_grad():
                predicted_actions = model(states_t, actions_t, rtg_t)
                action = predicted_actions[0, -1, 0].item()  # last timestep prediction

            # Step environment
            next_state, reward, done = env.step(action)

            # Save trajectory
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'rtg': rtg_history[-1]
            })

            # Update history
            states_history.append(next_state)
            actions_history.append(action)
            rtg_history.append(rtg_history[-1] - reward)  # update RTG

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_returns.append(episode_reward)
        all_trajectories.append(trajectory)
        print(f"Episode {ep+1}/{num_episodes}: Return = {episode_reward:.1f}, Steps = {len(trajectory)}")

    # Statistics
    returns_array = np.array(episode_returns)
    mean_return = returns_array.mean()
    std_return = returns_array.std()

    print(f"\n{'='*50}")
    print(f"Evaluation Results (target RTG={target_rtg})")
    print(f"{'='*50}")
    print(f"Mean Return: {mean_return:.1f} ± {std_return:.1f}")
    print(f"Min Return: {returns_array.min():.1f}")
    print(f"Max Return: {returns_array.max():.1f}")
    print(f"Success Rate (≥180): {(returns_array >= 180).mean()*100:.1f}%")

    # Plot RTG vs achieved return
    plot_rtg_vs_achieved(all_trajectories, episode_returns)

    # Render best episode if requested
    if render_best:
        best_idx = np.argmax(episode_returns)
        save_trajectory_gif(all_trajectories[best_idx], f"videos/best_episode_{returns_array[best_idx]:.0f}.png")

    return {
        'mean': mean_return,
        'std': std_return,
        'returns': episode_returns,
        'trajectories': all_trajectories
    }

def plot_rtg_vs_achieved(trajectories, returns):
    """Plot target RTG vs achieved return for all episodes"""
    import os
    os.makedirs('figs', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: RTG decay over time for each episode
    for i, traj in enumerate(trajectories[:5]):  # show first 5
        rtgs = [t['rtg'] for t in traj]
        ax1.plot(rtgs, alpha=0.6, label=f"Ep {i+1} (ret={returns[i]:.0f})")
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Return-to-Go')
    ax1.set_title('RTG Conditioning Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of returns
    ax2.hist(returns, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}')
    ax2.axvline(180, color='green', linestyle='--', label='Success threshold')
    ax2.set_xlabel('Episode Return')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figs/eval_results.png', dpi=150)
    print(f"✓ Saved plot to figs/eval_results.png")
    plt.close()

def save_trajectory_gif(trajectory, filename):
    """Save a visualization of the trajectory"""
    import os
    os.makedirs('videos', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    timesteps = range(len(trajectory))
    angles = [t['state'][0] for t in trajectory]
    angular_vels = [t['state'][1] for t in trajectory]
    positions = [t['state'][2] for t in trajectory]
    actions = [t['action'] for t in trajectory]

    axes[0, 0].plot(timesteps, angles)
    axes[0, 0].set_ylabel('Pole Angle (rad)')
    axes[0, 0].set_title('Pole Angle')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(timesteps, angular_vels)
    axes[0, 1].set_ylabel('Angular Velocity')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(timesteps, positions)
    axes[1, 0].set_ylabel('Cart Position')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_title('Cart Position')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(timesteps, actions)
    axes[1, 1].set_ylabel('Force')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_title('Action (Force)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✓ Saved trajectory to {filename}")
    plt.close()

if __name__ == "__main__":
    # Load trained model
    print("Loading model...")
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
    print("\nEvaluating policy...")
    results = eval_policy(
        model,
        target_rtg=200.0,
        context_len=20,
        num_episodes=20,
        device=device,
        render_best=True
    )
