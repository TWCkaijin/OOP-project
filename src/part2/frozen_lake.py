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
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        if os.path.exists('frozen_lake8x8.pkl'):
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        else:
            print("No trained model found! Starting with random Q-table.")
            q = np.zeros((env.observation_space.n, env.action_space.n))


    learning_rate_a = 0.1   # Should we use a stable decade learning rate such such as 0.1 + 0.9*np.exp(-i / (episode/10)) ?
    discount_factor_g = 0.99 # emphasising the future choice (0.9 -> 0.99)
    epsilon = 1             
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    best_success_rate = 0.0 # recording the best solution

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        steps = 0

        # SARSA action choice
        if is_training and rng.random() < epsilon:
             action = rng.choice([0, 1, 2, 3])
        else:
             action = np.argmax(q[state, :])

        while(not terminated and not truncated):
            steps += 1
            
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 1:
                pass 
            elif terminated and reward == 0:
                reward = -1 # penalty for falling into a hole


            if is_training and rng.random() < epsilon:
                next_action = rng.choice([0, 1, 2, 3])
            else:
                next_action = np.argmax(q[new_state, :])

            if is_training :
                # SARSA 
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * q[new_state, next_action] - q[state, action]
                )
                
                if terminated:
                    q[state, action] = q[state, action] + learning_rate_a * (
                        reward - q[state, action]
                    )
            
            state = new_state
            action = next_action
        
        if new_state == 63:
            rewards_per_episode[i] = 1

        if is_training:
            epsilon = 0.001 + 0.999 * np.cos(min(1,i / (episodes/1.2)) * (np.pi/2))

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
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to run') # 預設改為 10000
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()

    run(args.episodes, is_training=args.train, render=args.render)