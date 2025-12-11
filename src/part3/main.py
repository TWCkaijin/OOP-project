from controller import RobotController
import argparse
import gymnasium as gym
import warehouse_env  # Register the environment
import yaml

def load_config(config_path="src/part3/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def train_agent(config, override_episodes=None, override_opponent=None):
    """
    Train a PPO agent using configuration from YAML.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return None
    
    # Extract config
    env_conf = config['environment']
    train_conf = config['training']
    path_conf = config['paths']
    reward_conf = config.get('rewards', None)
    
    # Allow overrides
    episodes = override_episodes if override_episodes is not None else train_conf['episodes']
    enable_opponent = override_opponent if override_opponent is not None else env_conf['enable_opponent']
    save_path = path_conf['model_save_path']
    
    env = gym.make('warehouse-robot-v0', render_mode=None, 
                   enable_opponent=enable_opponent,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf)
    
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
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=float(train_conf['learning_rate']),
        n_steps=train_conf['n_steps'],
        batch_size=train_conf['batch_size'],
        n_epochs=train_conf['n_epochs'],
        gamma=train_conf['gamma'],
        ent_coef=train_conf['ent_coef'],
        clip_range=train_conf['clip_range'],
        device='auto',
        policy_kwargs=dict(
            net_arch=train_conf.get('network_arch', [64, 64])
        )
    )
    
    callback = EpisodeCallback(target_episodes=episodes)
    
    print(f"Training for {episodes} episodes... (Opponent: {enable_opponent})")
    print("-" * 50)
    
    model.learn(total_timesteps=episodes * 2000, callback=callback)
    
    print("-" * 50)
    print(f"Training complete! {callback.episode_count} episodes")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()
    return model

def evaluate_agent(config, override_episodes=None, override_render=True, override_opponent=None):
    """Evaluate a trained agent using config"""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return
    
    path_conf = config['paths']
    env_conf = config['environment']
    reward_conf = config.get('rewards', None)
    
    model_path = path_conf['model_save_path']
    episodes = override_episodes if override_episodes is not None else 5
    enable_opponent = override_opponent if override_opponent is not None else env_conf['enable_opponent']
    
    model = PPO.load(model_path)
    env = gym.make('warehouse-robot-v0', render_mode='human' if override_render else None, 
                   enable_opponent=enable_opponent,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf)
    
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
    parser.add_argument('--eval', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--no-opponent', action='store_true', help='Disable the opponent (override YAML)')
    
    args = parser.parse_args()
    override_opponent = False if args.no_opponent else None
    
    if args.train:
        train_agent(config, override_episodes=args.episodes, override_opponent=override_opponent)
    elif args.eval:
        evaluate_agent(config, override_episodes=args.episodes, override_render=args.render, override_opponent=override_opponent)
    else:
        controller = RobotController(episodes=args.episodes or 5, render=args.render)
        controller.run()
