import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse 
import os

def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.where(rewards_per_episode == 1)[0].shape[0]
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate


def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.random.uniform(low=0, high=0.01, size=(env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        if os.path.exists('frozen_lake8x8.pkl'):
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        else:
            print("No trained model found! Starting with random Q-table.")
            q = np.zeros((env.observation_space.n, env.action_space.n))


    learning_rate_a = 0.035   # Learning rate
    discount_factor_g = 0.995 # Discount factor
    epsilon = 1             
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    best_success_rate = 0.0 # recording the best solution

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            
            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                action = rng.choice([0, 1, 2, 3])
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 1:
                pass 
            # elif terminated and reward == 0:
            #     reward = -1 # penalty
            # elif reward == 0:
            #     reward = -0.001 # step penalty
            
            # Q-Learning Update
            if is_training:
                target = reward
                if not terminated:
                    target += discount_factor_g * np.max(q[new_state, :])
                
                # Learning rate decay
                current_lr = max(0.005, learning_rate_a * (1 - i/episodes))
                q[state, action] = q[state, action] + current_lr * (target - q[state, action])
            
            state = new_state
        
        # Check if goal reached (reward is 1 and terminated)
        # Note: In FrozenLake, reward is 1 only when reaching the goal.
        if reward == 1 and terminated:
            rewards_per_episode[i] = 1

        # Decay epsilon
        if is_training:
            epsilon = max(0, 1 - i / (episodes * 0.85))

        past_reward_100 = rewards_per_episode[max(0, i-100):(i+1)]
        current_success_rate = np.sum(past_reward_100==1)

        if is_training and i > 100 and current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            with open("frozen_lake8x8.pkl", "wb") as f:
                pickle.dump(q, f)
                

        print(f"Episode {i+1}/{episodes} - Epsilon: {epsilon:.3f} - Success Rate (L100): {current_success_rate:2}% - Best: {best_success_rate:2}%", end='\r')

    env.close()

    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.title('Frozen Lake 8x8 - Rewards over Episodes (Last 100)')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards')
        plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))
    else:
        print(f"\nTraining finished. Best success rate achieved: {best_success_rate}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Car Agent Runner")
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes to run') 
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()

    run(args.episodes, is_training=args.train, render=args.render)