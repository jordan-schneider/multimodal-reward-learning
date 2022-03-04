Gym3 provides a slightly different api than people are used to.
observe() returns the state, reward, and first value.

The way episode ends work is that you start at s_t, then you take a_t, and if s_{t+1} would be a terminal state, instead reset the environment (setting first_{t+1} to True) and s_{t+1} is the start state for the new episode. All of your actions matter and you never see your own terminal states.