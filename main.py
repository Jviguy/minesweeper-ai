import random
from agent import DQNAgent
from env import MineSweeperEnv

env = MineSweeperEnv()
state_size = env.observation_space
action_size = env.action_space
agent = DQNAgent(state_size, action_size, "Agent1")
episodes = 10000  # Number of episodes to train
batch_size = 1000  # Size of the batch sampled from the replay buffer
def play(a, s):
    action = a.act(s, env)
    next_state, reward, done, _ = env.step(action)
    if log:
        print("Action:", action, "Reward:", reward)
    a.memory.add(s, action, reward, next_state, done)
    s = next_state
    if done:
        a.update_target_model()
        if log:
            print("Squares Left: ", env.squares_left-40)
            print("reward: ", reward)
        return []
    return s
for e in range(episodes):
    state = env.reset()
    log = False
    if random.random() < 0.02:
        log = True
    if agent.memory.size() > batch_size and random.random() < 0.05:
        agent.replay(batch_size)
    for time in range(500):  # Replace 500 with maximum time step
        if log:
            print(env)
        state = play(agent, state)
        if log:
            print(env)
        if len(state) == 0:
            break
    print("Episode ", e + 1, "Fin")