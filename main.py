import random
from agent import DQNAgent
from env import MineSweeperEnv
env = MineSweeperEnv()
state_size = env.observation_space
action_size = env.action_space
agent = DQNAgent(state_size, action_size, "Agent1")
episodes = 10000  # Number of episodes to train
batch_size = 10000  # Size of the batch sampled from the replay buffer

for e in range(episodes):
    state = env.reset()
    log = False
    step = 0
    if random.random() < 0.02:
        log = True
    done = False
    while not done:  # Replace 500 with maximum time step
        action = agent.act(state, env)
        next_state, reward, done, _ = env.step(action)
        if log:
            print("Action:", action, "Reward:", reward)
        agent.update_replay_memory((state, action, reward, next_state, done))
        state = next_state
        agent.train(done, step)
        if done:
            if log:
                print("Squares Left: ", env.squares_left-40)
                print("reward: ", reward)
        if log:
            print(env)
    print("Episode ", e + 1, "Fin")
