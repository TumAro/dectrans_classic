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
