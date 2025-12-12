import argparse

import gymnasium as gym
import warehouse_env  # Register the environment
import yaml

def load_config(config_path="src/part3/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def train_agent(config, override_episodes=None, override_opponent=None, continue_training=False):
    """
    Train a PPO agent using configuration from YAML.
    """
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return None
    
    import os
    
    # Extract config
    env_conf = config['environment']
    train_conf = config['training']
    path_conf = config['paths']
    reward_conf = config.get('rewards', None)
    
    # Allow overrides
    save_path = path_conf['model_save_path']
    stage = env_conf.get('stage', 3)  # Default to 3 if not in config
    random_obstacles = env_conf.get('random_obstacles', False)
    
    env = gym.make('warehouse-robot-v0', render_mode=None, 
                   enable_opponent=enable_opponent,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   random_obstacles=random_obstacles,
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf,
                   stage=stage)
    
    # Custom callback to track episodes
    class EpisodeCallback(BaseCallback):
        def __init__(self, target_episodes, verbose=1):
            super().__init__(verbose)
            self.target_episodes = target_episodes
            self.episode_count = 0
            self.episode_rewards = []
            self.current_reward = 0
            
        def _on_step(self) -> bool:
            # Track reward
            self.current_reward += self.locals.get('rewards', [0])[0]
            
            # Check if episode ended
            dones = self.locals.get('dones', [False])
            if dones[0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0
                
                # Log progress every 100 episodes
                if self.episode_count % 100 == 0:
                    avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                    print(f"Episode {self.episode_count}/{self.target_episodes} | Avg Reward (last 100): {avg_reward:.2f}")
                
                # Stop if target reached
                if self.episode_count >= self.target_episodes:
                    return False
            return True
    
    algo_type = train_conf.get('algorithm', 'PPO')
    algo_params = train_conf.get(algo_type.lower(), {})
    device = train_conf.get('device', 'cpu')
    
    model = None
    
    # Check if we should continue training
    if continue_training:
        if os.path.exists(save_path):
            print(f"Loading existing model from {save_path} to continue training...")
            if algo_type == 'DQN':
                model = DQN.load(save_path, env=env, device=device)
            else:
                model = PPO.load(save_path, env=env, device=device)
        else:
            print(f"Warning: Model file {save_path} not found. Starting new training session.")

    # Convert to float manually to avoid YAML types issues
    learning_rate = float(algo_params.get('learning_rate', 3e-4 if algo_type == 'PPO' else 1e-4))

    if model is None:
        if algo_type == 'DQN':
            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=learning_rate,
                buffer_size=algo_params.get('buffer_size', 50000),
                learning_starts=algo_params.get('learning_starts', 1000),
                batch_size=algo_params.get('batch_size', 32),
                gamma=algo_params.get('gamma', 0.99),
                train_freq=algo_params.get('train_freq', 4),
                gradient_steps=algo_params.get('gradient_steps', 1),
                target_update_interval=algo_params.get('target_update_interval', 250),
                exploration_fraction=algo_params.get('exploration_fraction', 0.1),
                exploration_final_eps=algo_params.get('exploration_final_eps', 0.05),
                device=device,
                policy_kwargs=dict(
                    net_arch=algo_params.get('network_arch', [128, 128])
                )
            )
        else:
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=0,
                learning_rate=learning_rate,
                n_steps=algo_params.get('n_steps', 2048),
                batch_size=algo_params.get('batch_size', 64),
                n_epochs=algo_params.get('n_epochs', 10),
                gamma=algo_params.get('gamma', 0.99),
                ent_coef=algo_params.get('ent_coef', 0.01),
                clip_range=algo_params.get('clip_range', 0.2),
                device=device,
                policy_kwargs=dict(
                    net_arch=algo_params.get('network_arch', [64, 64])
                )
            )
    
    callback = EpisodeCallback(target_episodes=episodes)
    
    print(f"Training for {episodes} episodes... (Opponent: {enable_opponent})")
    print(f"Device: {device}")
    print("-" * 50)
    print("Configuration Summary:")
    print(f"  Algorithm: {algo_type}")
    print(f"  Environment: {env_conf}")
    print(f"  Rewards: {reward_conf}")
    print(f"  Training Params ({algo_type}):")
    for k, v in algo_params.items():
        print(f"    {k}: {v}")
    print("-" * 50)
    
    # reset_num_timesteps=False is important when continuing training to keep tensorboard logs consistent
    # ensuring continuity if using TB, though we aren't explicitly here.
    model.learn(total_timesteps=episodes * 2000, callback=callback, reset_num_timesteps=not continue_training)
    
    print("-" * 50)
    print(f"Training complete! {callback.episode_count} episodes")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()
    return model

def evaluate_agent(config, override_episodes=None, override_render=True, override_opponent=None):
    """Evaluate a trained agent using config"""
    try:
        from stable_baselines3 import PPO, DQN
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return
    
    path_conf = config['paths']
    env_conf = config['environment']
    reward_conf = config.get('rewards', None)
    train_conf = config.get('training', {})
    
    model_path = path_conf['model_save_path']
    episodes = override_episodes if override_episodes is not None else 5
    enable_opponent = override_opponent if override_opponent is not None else env_conf['enable_opponent']
    stage = env_conf.get('stage', 3)
    random_obstacles = env_conf.get('random_obstacles', False)
    
    algo_type = train_conf.get('algorithm', 'PPO')
    
    if algo_type == 'DQN':
        model = DQN.load(model_path, device=train_conf.get('device', 'cpu'))
    else:
        model = PPO.load(model_path, device=train_conf.get('device', 'cpu'))
    env = gym.make('warehouse-robot-v0', render_mode='human' if override_render else None, 
                   enable_opponent=enable_opponent,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   random_obstacles=random_obstacles,
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf,
                   stage=stage)
    
    total_rewards = []
    total_steps = []
    completed = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        steps = 0
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            
            if steps > 300:
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        if info.get('task_completed', False):
            completed += 1
            
        print(f"Episode {ep+1}: Reward={ep_reward:.2f}, Steps={steps}")
    
    print("-" * 50)
    print(f"Summary: {completed}/{episodes} completed")
    print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Avg Steps: {sum(total_steps)/len(total_steps):.1f}")
    
    env.close()

if __name__ == '__main__':
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Warehouse Robot Runner")
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes') 
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--train', action='store_true', help='Train a new agent')
    parser.add_argument('--continue-train', action='store_true', help='Continue training existing agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--no-opponent', action='store_true', help='Disable the opponent (override YAML)')
    
    args = parser.parse_args()
    override_opponent = False if args.no_opponent else None
    
    if args.train:
        train_agent(config, override_episodes=args.episodes, override_opponent=override_opponent, continue_training=False)
    elif args.continue_train:
         train_agent(config, override_episodes=args.episodes, override_opponent=override_opponent, continue_training=True)
    elif args.eval:
        evaluate_agent(config, override_episodes=args.episodes, override_render=args.render, override_opponent=override_opponent)
    else:
        print("Please specify --train, --continue-train, or --eval mode.")
