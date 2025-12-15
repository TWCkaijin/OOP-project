import gymnasium as gym
import time
import warehouse_env  # Register the environment

class RobotController:
    def __init__(self, episodes=1, render=True):
        self.episodes = episodes
        self.render = render
        self.env_id = 'warehouse-robot-v0'
        
    def _get_heuristic_action(self, obs):
        """
        Simple heuristic for multi-trip delivery:
        obs: [row, col, carrying, delivered, total, dx, dy, remaining, ...]
        
        Strategy:
        - If carrying < max and targets remain: go to nearest cargo
        - If carrying == max or no targets: return to origin
        """
        carrying = obs[2]
        remaining = obs[7]
        dx = obs[5]
        dy = obs[6]
        
        # dx, dy already point to the right target (cargo or origin)
        # based on env's _get_nearest_target_info logic
        
        # Move along largest distance axis
        if dy < 0: return 3  # UP
        elif dy > 0: return 1  # DOWN
        elif dx < 0: return 0  # LEFT
        elif dx > 0: return 2  # RIGHT
        
        return 0  # At target

    def run(self):
        try:
            env = gym.make(self.env_id, render_mode='human' if self.render else None)
        except Exception as e:
            print(f"Error creating environment '{self.env_id}': {e}")
            return

        print(f"Controller started: Running {self.episodes} episodes.")

        for i in range(self.episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            step_count = 0
            total_reward = 0
            
            print(f"\nEpisode {i+1}: {info['total_cargos']} cargos to deliver")

            while not terminated and not truncated:
                action = self._get_heuristic_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                if self.render:
                    time.sleep(0.15)
                
                # Safety break
                if step_count > 300:
                    truncated = True
            
            status = "✓ Completed" if info.get('task_completed') else "✗ Timeout"
            print(f"Episode {i+1} {status}. Delivered: {info['delivered']}/{info['total_cargos']}, Steps: {step_count}, Reward: {total_reward:.2f}")

        env.close()
