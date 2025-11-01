# dectrans_classic

## Journal

### Setting Up Process
I am going to create a small AI that learns to balance a CartPole without use of traditional RL.

First I have to define the physics of CartPOle system. Realise that the constants for our system are,
- $l \rightarrow$ length of pole
- $m \rightarrow$ mass of pole
- $M \rightarrow$ mass of cart
- $g \rightarrow$ gravitational constant $9.81 m/s^2$

And the variables of our system are,
- $\theta \rightarrow$ tilt angle of the pole
- $\dot{\theta} \rightarrow$ angular velocity of the pole
- $x \rightarrow$ the position of cart
- $\dot{x} \rightarrow$ the velocity of the cart

![cartpole system](figs/cartpole.png)

After applying the lagrangian and solving for the motion of equation (which I ain't gonna do myself right now) we will get the equations realising here $x$ is a *free* variable which is true as where the pole is moving to seldom depends upon where the cart is but on how fast the cart is moving. [source](https://in.mathworks.com/help/symbolic/derive-and-simulate-cart-pole-system.html)

Also we notice the equation of motion for the pole and the cart are coupled and interdependent on eachother because when cart is moving it asserts some force on the pole which consequently because of newton's second law of motion exerts a friction and yanks the cart in return.

So in our `step(u)` code where $u$ is the external control force, we implement semi-implicit Euler step function, ie, updating the velocity first then updating the position.
```
velocity_new = velocity_old + acceleration * dt
position_new = position_old + velocity_new * dt
```

Now even if there is the utopian assumption that there is no friction or air drag in simulation there are other issues which may occur like, the euler step may acquire soem drift over steps due to computational constraints. So we make sure to add tests with pytest. AND for my test it worked beautifully fine!

And finally I personally test something, I implement random pushes and shoves throughout the timesteps uniformly to check if everything works perfectly or not. IF the physics work correctly
- The movement will be smooth in all case and never jagged
- Does the system reach boundaries of threshold we set
- Does it or not oscillate or diverge?
- Are there randomness in mtrajectory given we put random forces?

![theta trajectory](figs/theta_trajectory.png)

AAND I am proud to say it worked absolutely greatly for my code!
Now we can start with the next step!
