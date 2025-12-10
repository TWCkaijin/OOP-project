import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

class DistanceRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.95):
        super().__init__(env)
        self.gamma = gamma
        
        # use the unwrapped FrozenLake environment
        desc = self.env.unwrapped.desc
        self.n = desc.shape[0]
        
        # define goal (bottom-right corner)
        self.goal_pos = (self.n - 1, self.n - 1)

    # convert state index → (row, col)
    def state_to_pos(self, state):
        return (state // self.n, state % self.n)

    def manhattan_distance(self, pos):
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def step(self, action):
        old_state = self.env.unwrapped.s
        new_state, reward, terminated, truncated, info = self.env.step(action)

        if terminated and reward > 0:
            # reached goal
            return new_state, 10, terminated, truncated, info
        if terminated and reward == 0:
            # fell in hole
            return new_state, -10, terminated, truncated, info
        
        
        old_pos = self.state_to_pos(old_state)
        new_pos = self.state_to_pos(new_state)

        old_dist = self.manhattan_distance(old_pos)
        new_dist = self.manhattan_distance(new_pos)

        # potential-based shaping
        shaped_reward = old_dist - new_dist
        return new_state, shaped_reward, terminated, truncated, info



def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, success_rate=0.75, render_mode='ansi' if render else None)
    env = DistanceRewardWrapper(env)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
        epsilon = 1
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        epsilon = 0.04
        f.close()

    learning_rate_a = 0.999 # alpha or learning rate
    lr_decay_rate = 0.999
    lr_min = 0.01
    discount_factor_g = 0.95 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    min_exploration_rate = 0.04
    epsilon_decay_rate = (epsilon - min_exploration_rate) / (episodes * 0.95)
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    running_success_rate = []

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        if(epsilon==0):
            learning_rate_a = 0.0001
        
        learning_rate_a = max(learning_rate_a * lr_decay_rate, lr_min)

        if reward >= 10:
            rewards_per_episode[i] = 1

        running_success_rate.append(np.sum(rewards_per_episode[max(0, i-100):(i+1)]) / min(100, i+1) * 100)
        print(f"Episode {i+1}/{episodes} | Epsilon: {epsilon:.4f} | Success Rate (last 100): {running_success_rate[-1]:.2f}%")

    env.close()

    # plot sum of rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    plt.clf()
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Car Agent Runner")
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes to run') 
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()

    print(args.train)

    run(args.episodes, is_training=args.train, render=args.render)

    # run(15000, is_training=True, render=False)
    # run(1000, is_training=False, render=True)