from src.envs.cartpole import CartPole


def zeroForce(steps=100):

    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)
    state = env.reset()
    print(f"Initial state: {state}")
    
    for i in range(steps):
        state, reward, done = env.step(u=0)
        print(f"Step {i}|  theta={state[0]:7.4f}, x={state[2]:7.4f}, reward={reward}, done={done}")
        if done:
            print(f"Episode ended at step {i}")
            break