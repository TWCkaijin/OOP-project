import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import sys
import os

# Ensure we can import the local environment module
# This allows running from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import oop_project_env

def run(episodes, is_training=True, render=False):
    env = gym.make('warehouse-robot-v0', render_mode='human' if render else None)

    # State space dimensions: Robot Rows (4) x Robot Cols (5) x Target Rows (4) x Target Cols (5)
    # Action space: 4
    # We can fetch grid size from env.unwrapped if needed, but let's assume default 4x5 for now or extra safety
    rows = env.unwrapped.grid_rows
    cols = env.unwrapped.grid_cols
    
    # Q-table shape: (rows, cols, rows, cols, actions)
    q_shape = (rows, cols, rows, cols, env.action_space.n)

    if is_training:
        q = np.zeros(q_shape)
    else:
        model_path = os.path.join(os.path.dirname(__file__), 'warehouse_q_table.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                q = pickle.load(f)
        else:
            print("No trained model found! Starting with random/empty Q-table.")
            q = np.zeros(q_shape)

    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 1.0 / (episodes * 0.8) if episodes > 0 else 0
    min_epsilon = 0.01

    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    # Track success (reaching target)
    # Reward is 1 only on success
    
    for i in range(episodes):
        obs, _ = env.reset()
        # obs is [rr, rc, tr, tc]
        rr, rc, tr, tc = obs
        
        terminated = False
        truncated = False
        steps = 0
        max_steps = 100 # Prevent infinite loops

        rewards = 0

        while not terminated and not truncated and steps < max_steps:
            steps += 1
            
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Argmax over the last dimension for the specific state
                action = np.argmax(q[rr, rc, tr, tc, :])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            n_rr, n_rc, n_tr, n_tc = next_obs
            
            # Additional penalty for taking too long to encourage shortest path?
            # The env gives 0 for steps, 1 for goal.
            # Standard Q-learning works with 0 step reward, but -0.01 helps convergence usually.
            # But let's stick to env rewards. 
            
            if is_training:
                best_next_action = np.argmax(q[n_rr, n_rc, n_tr, n_tc, :])
                td_target = reward + discount_factor * q[n_rr, n_rc, n_tr, n_tc, best_next_action]
                td_error = td_target - q[rr, rc, tr, tc, action]
                q[rr, rc, tr, tc, action] += learning_rate * td_error

            rr, rc, tr, tc = n_rr, n_rc, n_tr, n_tc
            rewards += reward

        rewards_per_episode[i] = rewards
        
        if is_training:
            epsilon = max(min_epsilon, epsilon - epsilon_decay)

        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{episodes} - Epsilon: {epsilon:.3f}", end='\r')

    env.close()

    if is_training:
        model_path = os.path.join(os.path.dirname(__file__), 'warehouse_q_table.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(q, f)
        
        # Plotting
        mean_rewards = np.zeros(episodes)
        for t in range(episodes):
            mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
        
        plt.figure()
        plt.plot(mean_rewards)
        plt.title('Warehouse Robot - Success Rate (Last 100)')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Reward / Success Rate')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'warehouse_training.png'))
        print("\nTraining completed. Model saved.")

    else:
        success_rate = np.mean(rewards_per_episode) * 100
        print(f"\nEvaluation completed. Success Rate: {success_rate:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Warehouse Agent Runner")
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()
    run(args.episodes, is_training=args.train, render=args.render)
