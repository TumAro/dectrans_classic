from src.envs.cartpole import CartPole
import random
import matplotlib.pyplot as plt

def randomForce(episodes=10, max_steps=500):
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)

    # storing theta
    all_theta_data = []

    for episode in range(episodes):
        state = env.reset()
        theta_history = []

        # print(f"Initial state: {state}")
        
        for step in range(max_steps):
            u = random.uniform(-10, 10) #random force between -10N and +10N
            state, reward, done = env.step(u)

            theta_history.append(state[0])
            if done:
                print(f"Episode {episode} : ended at step {step}")
                break

        all_theta_data.append(theta_history)

    for i, theta_history in enumerate(all_theta_data):
        plt.plot(theta_history, label=f"Episode {i}")

    plt.xlabel('timestep')
    plt.ylabel('theta')
    plt.legend()
    plt.show()