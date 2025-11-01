from src.envs.cartpole import CartPole

def test_reset_cart_at_origin():
    """Test that reset() put cart at x=0"""
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)
    state = env.reset()

    assert state[2] == 0.0, "Cart should start at x=0"
    assert state[3] == 0.0, "Cart shoould start with zero velocity"

import math
def calculate_total_energy(env: CartPole):
    theta, theta_dot, x, x_dot = env.state

    # * KINETIC ENERGY
    KE_c = 0.5 * env._M * x_dot**2
    KE_p = 0.5 * (1/3) * env._m * env._l**2 * theta_dot**2

    # * POTENTIAL ENERGY
    PE_p = env._m * env._g * env._l * math.cos(theta)

    return KE_c + KE_p + PE_p

def test_energy_explosion():
    """Test that total energy stays bounded with small timestep"""
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)
    state = env.reset()

    E_initial = calculate_total_energy(env)

    for _ in range(1000):
        _, _, done = env.step(u=0.0)
        if done:
            break

    E_final = calculate_total_energy(env)

    energy_ratio = abs(E_final / E_initial)
    assert energy_ratio < 10.0, f"Energy exploded! Initial: {E_initial:.4f}, Final: {E_final:.4f}"
    print(f"|| Energy stayed bounded: {E_initial:.4f} --> {E_final:.4f} (ratio: {energy_ratio:.2f})")

def test_pole_falls_without_input():
    """Test that pole falls when no force is applied"""
    env = CartPole(pole_mass=0.1, cart_mass=1.0, pole_length=0.5)
    state = env.reset()

    initial_theta = env._theta

    for _ in range(100):
        state, reward, done = env.step(u=0.0)
        if done:
            break

    final_theta = state[0]

    assert abs(final_theta) > abs(initial_theta), "Pole should tilt more under gravity"
    assert done == True, "Episode should end when pole falls"

    print(f"|| Pole fell: {initial_theta:.4f} --> {final_theta:.4f}")