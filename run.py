import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")

env.reset()

for _ in range(5):
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        if done: break

env.close()

