## state[tilt, angular velocity, cart velocity, cart acceleration]

import math

class CartPole:
    def __init__(self, pole_mass, cart_mass, pole_length, dt = 0.02):
        self._m = pole_mass
        self._l = pole_length
        self._M = cart_mass

        # kinematics data
        self._g = 9.81
        self._dt = dt

        self._theta = 0.01
        self._theta_d = 0.0

        self._x = 0.00
        self._x_dot = 0.0

    @property
    def state(self):
        return [self._theta, self._theta_d, self._x, self._x_dot]

    def reset(self, seed=None):
        self._theta = 0.01
        self._theta_d = 0.0
        self._x = 0.0
        self._x_dot = 0.0

        return self.state

    def step(self, u):
        reward = 0

        cos = math.cos(self._theta)
        sin = math.sin(self._theta)

        temp = (u + self._m * self._l * self._theta_d**2 * sin) / (self._M + self._m)
        # angular acceleration of pole
        theta_ddot = (self._g*sin - temp*cos) / (self._l * (4/3 - self._m * cos**2 / (self._M + self._m)))
        # linear acceleration of cart
        x_ddot = temp - self._m * self._l * theta_ddot * cos / (self._M + self._m)

        # updating velocities
        self._x_dot = self._x_dot + x_ddot*self._dt
        self._theta_d = self._theta_d + theta_ddot*self._dt

        # updating positions
        self._x = self._x + self._x_dot*self._dt
        self._theta = self._theta + self._theta_d*self._dt

        done = abs(self._theta) > 0.3 or abs(self._x) >= 2.4

        if not done:
            reward = 1

        return (self.state, reward, done)