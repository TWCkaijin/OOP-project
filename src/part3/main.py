import argparse

import gymnasium as gym
import warehouse_env  # Register the environment
import yaml
import numpy as np
import os

def load_config(config_path="src/part3/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_opponent_model(config, agent_id, device):
    """
    Load opponent model for iterative training.
    """
    from stable_baselines3 import PPO, DQN
    
    path_conf = config['paths']
    train_conf = config['training']
    
    opp_id = 1 - agent_id
    opp_path = path_conf['model_save_path']
    # Enforce p1/p2 suffixes
    if opp_id == 1:
        opp_path += "_p2"
    elif opp_id == 0:
        opp_path += "_p1"
        
    if os.path.exists(opp_path + ".zip") or os.path.exists(opp_path):
         print(f"Loading opponent model (Agent {opp_id}) from {opp_path}...")
         algo_type = train_conf.get('algorithm', 'PPO')
         if algo_type == 'DQN':
             return DQN.load(opp_path, device=device)
         else:
             return PPO.load(opp_path, device=device)
    else:
         print(f"No trained opponent found at {opp_path}. Using Greedy heuristic.")
         return None

def train_agent(config, override_episodes=None, override_opponent=None, continue_training=False, agent_id=0):
    """
    Train a PPO agent using configuration from YAML.
    """
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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
    if agent_id == 1:
        save_path += "_p2"
    elif agent_id == 0:
        save_path += "_p1"
        
    stage = env_conf.get('stage', 3)  # Default to 3 if not in config
    random_obstacles = env_conf.get('random_obstacles', False)
    
    # Try to load opponent model for iterative training
    opponent_model = None
    if override_opponent is not False:
        device = train_conf.get('device', 'cpu')
        opponent_model = get_opponent_model(config, agent_id, device)
    
    env = gym.make('warehouse-robot-v0', render_mode=None, 
                   enable_opponent=override_opponent,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   random_obstacles=random_obstacles,
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf,
                   stage=stage,
                   agent_id=agent_id,
                   opponent_model=opponent_model)
    
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
        # Check for .zip extension handling in SB3
        check_path = save_path
        if not os.path.exists(check_path) and os.path.exists(check_path + ".zip"):
            check_path = check_path + ".zip"
            
        if os.path.exists(check_path):
            print(f"Loading existing model from {check_path} to continue training...")
            if algo_type == 'DQN':
                model = DQN.load(check_path, env=env, device=device)
            else:
                model = PPO.load(check_path, env=env, device=device)
        else:
            print(f"Warning: Model file {save_path} (or .zip) not found. Starting new training session.")

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
    
    callback = EpisodeCallback(target_episodes=train_conf.get('episodes', override_episodes))
    
    print(f"Training for {train_conf.get('episodes', override_episodes)} episodes... (Opponent: {override_opponent})")
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
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=train_conf.get('episodes', override_episodes) * 2000/20, 
        save_path=os.path.dirname(save_path) + "/checkpoints",
        name_prefix="ckpt_" + algo_type.lower()
    )
    
    callbacks = [callback, checkpoint_callback]
    
    try:
        # reset_num_timesteps=False is important when continuing training to keep tensorboard logs consistent
        model.learn(total_timesteps=train_conf.get('episodes', 100000)*2000, callback=callbacks, reset_num_timesteps=not continue_training)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save(save_path)
        print(f"Model saved to {save_path}")
        env.close()
        return model
    
    print("-" * 50)
    print(f"Training complete! {callback.episode_count} episodes")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()
    return model

def evaluate_agent(config, override_episodes=None, override_render=True, override_opponent=None, agent_id=0):
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
    if agent_id == 1:
        model_path += "_p2"
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
                   stage=stage,
                   agent_id=agent_id)
    
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
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes}: Reward={ep_reward:.2f}, Steps={steps}")
    
    success_rate = (completed / episodes) * 100
    print("-" * 50)
    print(f"Summary: {completed}/{episodes} completed")
    print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Avg Steps: {sum(total_steps)/len(total_steps):.1f}")
    print(f"Evaluation completed. Success Rate: {success_rate:.2f}%")
    
    env.close()

def run_battle(config, override_episodes=None, override_render=True):
    """
    Run a battle between two trained agents (Agent 1 vs Agent 2).
    Requires both models to be trained and saved.
    """
    try:
        from stable_baselines3 import PPO, DQN
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return
    
    import os

    path_conf = config['paths']
    env_conf = config['environment']
    reward_conf = config.get('rewards', None)
    train_conf = config.get('training', {})
    
    # Paths
    path1 = path_conf['model_save_path']
    path2 = path_conf['model_save_path'] + "_p2"
    
    # Check if models exist
    if not (os.path.exists(path1 + ".zip") or os.path.exists(path1)):
        print(f"Error: Agent 1 model not found at {path1}")
        return
    if not (os.path.exists(path2 + ".zip") or os.path.exists(path2)):
        print(f"Error: Agent 2 model not found at {path2}")
        return
        
    print(f"Loading Agent 1 (Top-Left) from {path1}...")
    print(f"Loading Agent 2 (Bottom-Right) from {path2}...")
    
    device = train_conf.get('device', 'cpu')
    algo_type = train_conf.get('algorithm', 'PPO')
    
    # Load Models
    if algo_type == 'DQN':
        model1 = DQN.load(path1, device=device)
        model2 = DQN.load(path2, device=device)
    else:
        model1 = PPO.load(path1, device=device)
        model2 = PPO.load(path2, device=device)
        
    episodes = override_episodes if override_episodes is not None else 5
    stage = env_conf.get('stage', 3)
    # Default to False for fair battle unless specified
    random_obstacles = env_conf.get('random_obstacles', False)
    
    # Create Env (Agent 0 as primary controller)
    env = gym.make('warehouse-robot-v0', render_mode='human' if override_render else None, 
                   enable_opponent=True,
                   enable_obstacles=env_conf.get('enable_obstacles', True),
                   random_obstacles=random_obstacles,
                   max_steps=env_conf.get('max_steps', 1000),
                   reward_config=reward_conf,
                   stage=stage,
                   agent_id=0) 
    
    p1_wins = 0
    p2_wins = 0
    
    for ep in range(episodes):
        obs1, info = env.reset() # Reset returns obs for agent_id=0
        
        terminated = False
        truncated = False
        steps = 0
        
        print(f"\n--- Battle Episode {ep+1} ---")
        
        while not terminated and not truncated:
            # 1. Get Action for Agent 1
            action1, _ = model1.predict(obs1, deterministic=True)
            
            # 2. Get Action for Agent 2
            # Switch perspective temporarily to get Obs for Agent 2
            # Use unwrapped to access internal methods/attributes blocked by Gym wrappers
            unwrapped_env = env.unwrapped
            unwrapped_env.agent_id = 1 
            obs2 = unwrapped_env._get_obs()
            unwrapped_env.agent_id = 0 # Switch back
            
            action2, _ = model2.predict(obs2, deterministic=True)
            
            # 3. Step Environment
            # Use unwrapped step to allow passing opponent_action
            obs1, reward, terminated, truncated, info = unwrapped_env.step(action1, opponent_action=action2)
            
            steps += 1
            if steps > 500: # Safety break
                truncated = True
        
        # Result
        # For battle, delivered count is the key metric
        # Access robot instances directly
        p1_score = unwrapped_env.robot.delivered_count
        p2_score = unwrapped_env.robot.robot2_delivered
        
        print(f"Result: P1: {p1_score} | P2: {p2_score}")
        
        if p1_score > p2_score:
            p1_wins += 1
            print("Winner: Agent 1")
        elif p2_score > p1_score:
            p2_wins += 1
            print("Winner: Agent 2")
        else:
            print("Draw")
            
    print("-" * 30)
    print(f"Battle Summary ({episodes} games):")
    print(f"Agent 1 Wins: {p1_wins}")
    print(f"Agent 2 Wins: {p2_wins}")
    env.close()

if __name__ == '__main__':
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Warehouse Robot Runner")
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes') 
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--train', action='store_true', help='Train a new agent')
    parser.add_argument('--continue-train', action='store_true', help='Continue training existing agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--battle', action='store_true', help='Run 1v1 Battle between trained agents')
    parser.add_argument('--no-opponent', action='store_true', help='Disable the opponent (override YAML)')
    
    parser.add_argument('--agent-id', type=int, default=0, help='Agent ID to train (0: Top-Left, 1: Bottom-Right)')
    
    args = parser.parse_args()
    override_opponent = False if args.no_opponent else None
    
    if args.train:
        train_agent(config, override_episodes=args.episodes, override_opponent=override_opponent, continue_training=False, agent_id=args.agent_id)
    elif args.continue_train:
         train_agent(config, override_episodes=args.episodes, override_opponent=override_opponent, continue_training=True, agent_id=args.agent_id)
    elif args.eval:
        evaluate_agent(config, override_episodes=args.episodes, override_render=args.render, override_opponent=override_opponent, agent_id=args.agent_id)
    elif args.battle:
        run_battle(config, override_episodes=args.episodes, override_render=args.render)
    else:
        print("Please specify --train, --continue-train, --eval, or --battle mode.")
