from src.envs.cartpole import CartPole
from src.transformer import heuristic_policy
import random
import matplotlib.pyplot as plt

def heuristicForce(K=10, max_steps=500):
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)

    # storing theta
    state = env.reset()
    theta_history = []

    # print(f"Initial state: {state}")
    
    for step in range(max_steps):
        u = heuristic_policy(state, K)
        state, reward, done = env.step(u)

        theta_history.append(state[0])
        if done:
            break

    plt.plot(theta_history)

    plt.xlabel('timestep')
    plt.ylabel('theta')
    plt.legend()
    plt.show()