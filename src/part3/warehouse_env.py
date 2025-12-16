'''
Warehouse Robot Gym Environment
Handles: Environment configuration, reward calculation, observation space
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import warehouse_robot as wr
import numpy as np
import random
from obstacles import (ObstacleGenerator, FixedObstacleGenerator, RandomObstacleGenerator, EmptyObstacleGenerator)
from rewards import RewardStrategy, BasicReward, ShapingReward, CompetitiveReward
from robots import BaseRobot, GreedyRobot, RandomRobot, PatrolRobot

register(
    id='warehouse-robot-v0',
    entry_point='warehouse_env:WarehouseRobotEnv',
)

class AgentAdapter:
    """
    Encapsulates access to specific robot instances (Robot 1 or Robot 2)
    within the WarehouseRobot simulation.
    """
    def __init__(self, warehouse_robot, agent_id):
        self.wr = warehouse_robot
        self.id = agent_id
        
    @property
    def pos(self):
        return self.wr.robot_pos if self.id == 0 else self.wr.robot2_pos

    @property
    def start_pos(self):
        return [0, 0] if self.id == 0 else self.wr.robot2_start
        
    @property
    def carrying(self):
        return self.wr.carrying if self.id == 0 else self.wr.robot2_carrying
        
    @property
    def delivered_count(self):
        return self.wr.delivered_count if self.id == 0 else self.wr.robot2_delivered
    
    def move(self, action):
        if self.id == 0:
            return self.wr.move(action)
        else:
            return self.wr.move_bot2(action)

    def can_pick_more(self):
        return self.carrying < self.wr.max_carry


class WarehouseRobotEnv(gym.Env):
    """
    Multi-trip delivery environment:
    - Random 1-8 cargos spawn each episode
    - Robot can carry max 3 at a time
    - Must return to origin to deliver
    - Episode ends when all cargos delivered
    - Reward based on total steps (fewer = better)
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 16}

    def __init__(self, grid_rows=8, grid_cols=8, render_mode=None, 
                 min_cargos=1, max_cargos=8, max_carry=3, enable_opponent=True,
                 enable_obstacles=True, random_obstacles=False, max_steps=1000, reward_config=None, stage=3, **kwargs):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.stage = stage
        
        # Override settings based on stage
        if self.stage == 1:
            # Stage 1: Navigation Only
            # Single target, single carry, no opponent
            self.min_cargos = 1
            self.max_cargos = 1
            self.max_carry = 1
            self.enable_opponent = False
        elif self.stage == 2:
            # Stage 2: Cargo Pickup w/o Opponent
            # Full cargo logic, but no rival
            self.min_cargos = min_cargos
            self.max_cargos = max_cargos
            self.max_carry = max_carry
            self.enable_opponent = False
        else: # Stage 3 (Default/Full)
            # Stage 3: Competition
            self.min_cargos = min_cargos
            self.max_cargos = max_cargos
            self.max_carry = max_carry
            self.enable_opponent = enable_opponent # Use passed config or default True

        self.max_steps = max_steps
        
        # Reward strategy 
        self.rewards = reward_config if reward_config else {
            "step_penalty": -0.02,
            "collision_penalty": -1.0,
            "terminate_on_collision": True,
            "pickup_reward": 0.5,
            "delivery_base": 1.0,
            "delivery_combo": 0.5,
            "shaping_factor": 0.05,
            "efficiency_bonus": 0.2
        }
        
        # Select reward strategy based on config
        if self.enable_opponent:
            self.reward_strategy = CompetitiveReward(self.rewards)
        else:
            self.reward_strategy = ShapingReward(self.rewards)

        self.agent_id = kwargs.get('agent_id', 0) # 0: Top-Left (Main), 1: Bottom-Right (Opponent)
        self.opponent_model = kwargs.get('opponent_model', None) # Trained model for opponent

        # Create main robot 
        self.robot = wr.WarehouseRobot(
            grid_rows=grid_rows, 
            grid_cols=grid_cols, 
            fps=self.metadata['render_fps'],
            max_carry=self.max_carry,
            enable_opponent=self.enable_opponent
        )
        
        # Create Agent Adapters for cleaner logic
        self.agent = AgentAdapter(self.robot, self.agent_id)
        self.opponent = AgentAdapter(self.robot, 1 - self.agent_id)
        
        # Create opponent robot using polymorphic class (fallback if no model)
        if self.enable_opponent:
            if self.agent_id == 0:
                # We control Agent 1, Opponent is Agent 2 (Bottom-Right)
                opponent_start = [grid_rows - 1, grid_cols - 1]
                self.opponent_robot = GreedyRobot(opponent_start, grid_rows, grid_cols, self.max_carry)
            else:
                # We control Agent 2, Opponent is Agent 1 (Top-Left)
                opponent_start = [0, 0]
                self.opponent_robot = GreedyRobot(opponent_start, grid_rows, grid_cols, self.max_carry)
        else:
            self.opponent_robot = None

        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Observation: 
        # 1. robot_r / (rows-1)  [Normalized 0-1]
        # 2. robot_c / (cols-1)  [Normalized 0-1]
        # 3. carrying / max_carry [Normalized 0-1]
        # 4. Combined Grid Map (Flattened 8x8 = 64)
        #    Values: 0=Empty, 1=Cargo, -1=Obstacle, -2=Opponent
        
        base_obs_size = 5 # 3 original + 2 direction hints
        # Removed explicit opponent pos because it's in the grid now
        
        target_grid_size = self.grid_rows * self.grid_cols  # 64
        obs_size = base_obs_size + target_grid_size
        
        # Low values - Normalized Range [0.0, 1.0] for first 3, [-1.0, 1.0] for dir
        # Grid values remain -2, -1, 0, 1
        low = [0.0, 0.0, 0.0, -1.0, -1.0] 
        low.extend([-2.0] * target_grid_size) 
        
        # High values
        high = [1.0, 1.0, 1.0, 1.0, 1.0]
        high.extend([1.0] * target_grid_size)
        
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Reward parameters
        self.max_reward = 100.0
        # Internal scaler to keep rewards roughly around -1 to +1 range for PPO stability
        self.reward_scaler = 0.1
        
        # Environment state
        self.total_cargos = 0
        self.initial_targets = []
        
        # Obstacle Configuration - uses polymorphic generators
        self.enable_obstacles = enable_obstacles
        self.random_obstacles = random_obstacles
        
        # Select appropriate obstacle generator based on config
        self.agent_id = getattr(self, "agent_id", 0) # Default to 0 if not set (helper for below)

        # Select obstacle generator based on config
        if not self.enable_obstacles:
            self.obstacle_generator = EmptyObstacleGenerator(grid_rows, grid_cols)
        elif self.random_obstacles:
            self.obstacle_generator = RandomObstacleGenerator(grid_rows, grid_cols)
        else:
            self.obstacle_generator = FixedObstacleGenerator(grid_rows, grid_cols)
        
        self.obstacles = []  # populated in reset()

    def _spawn_cargos(self):
        """Spawn random 1-8 cargo targets"""
        self.total_cargos = random.randint(self.min_cargos, self.max_cargos)
        targets = []
        for _ in range(self.total_cargos):
            attempts = 0
            while attempts < 100:
                candidate = [
                    random.randint(0, self.grid_rows-1),
                    random.randint(0, self.grid_cols-1)
                ]
                if (candidate != [0, 0] and 
                    candidate not in targets and 
                    candidate not in self.obstacles):
                    targets.append(candidate)
                    break
                attempts += 1
        self.initial_targets = targets.copy()
        return targets

    def _get_opponent_action(self):
        """Get opponent action using trained model if available, else heuristic"""
        if self.opponent_model:
            # Swap identity to get opponent's observation
            # We swap the adapters so _get_obs() sees the opponent as "self"
            self.agent, self.opponent = self.opponent, self.agent
            
            try:
                obs = self._get_obs()
                action, _ = self.opponent_model.predict(obs, deterministic=True)
                return wr.RobotAction(int(action))
            finally:
                # Always swap back
                self.agent, self.opponent = self.opponent, self.agent

        # Fallback to heuristic robot
        # Opponent robot needs current state of the opponent agent
        self.opponent_robot.pos = list(self.opponent.pos)
        self.opponent_robot.carrying = self.opponent.carrying
            
        return self.opponent_robot.get_action(self.robot.targets, self.obstacles)

    def _bfs_distance(self, start, end):
        """Calculate distance (Manhattan for performance)"""
        # BFS is too slow for 100k+ steps training on Python
        # Manhattan is sufficient for reward shaping in this grid
        return abs(start[0]-end[0]) + abs(start[1]-end[1])

    def _get_nearest_target_info(self):
        """Get direction to optimal target using BFS distance"""
        # Determine self state via adapter
        robot_pos = self.agent.pos
        carrying = self.agent.carrying
        start_pos = self.agent.start_pos
        targets = self.robot.targets # Shared targets

        # If carrying max or no targets left, go to origin
        can_pick = carrying < self.max_carry
        if not targets or not can_pick:
            dest = start_pos
            dist = self._bfs_distance(robot_pos, dest)
            
            # Direction for info (kept simple Manhattan approx for vector)
            dy = dest[0] - robot_pos[0]
            dx = dest[1] - robot_pos[1]
            return dx, dy, dist
        
        # Find nearest cargo by TRUE distance
        min_dist = float('inf')
        nearest_diff = [0, 0]
        
        for t in targets:
            # Use BFS distance instead of Manhattan
            dist = self._bfs_distance(robot_pos, t)
            
            if dist < min_dist:
                min_dist = dist
                nearest_diff = [t[1] - robot_pos[1], t[0] - robot_pos[0]]
                
        return nearest_diff[0], nearest_diff[1], min_dist
            
        # Determine action to move towards dest
        dy = dest[0] - bot2_pos[0]
        dx = dest[1] - bot2_pos[1]
        
        if abs(dy) > abs(dx):
            return wr.RobotAction.DOWN if dy > 0 else wr.RobotAction.UP
        else:
            return wr.RobotAction.RIGHT if dx > 0 else wr.RobotAction.LEFT

    def _get_obs(self):
        # Determine self state
        robot_pos = self.agent.pos
        carrying = self.agent.carrying
        
        # Normalize continuous inputs
        norm_r = robot_pos[0] / (self.grid_rows - 1)
        norm_c = robot_pos[1] / (self.grid_cols - 1)
        norm_carry = carrying / self.max_carry
        
        obs_list = [norm_r, norm_c, norm_carry]
        
        # Add BFS direction hint to observation
        bfs_dx, bfs_dy, _ = self._get_nearest_target_info()
        mag = abs(bfs_dx) + abs(bfs_dy)
        if mag > 0:
            obs_list.append(bfs_dx / mag)
            obs_list.append(bfs_dy / mag)
        else:
            obs_list.extend([0.0, 0.0])
        
        # Combined Grid Map
        # 0: Empty
        # 1: Target
        # -1: Obstacle
        # -2: Opponent (Relative to whoever is training)
        grid_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Mark Obstacles
        for obs in self.obstacles:
            grid_map[obs[0], obs[1]] = -1.0
            
        # Mark Targets
        for t in self.robot.targets:
            grid_map[t[0], t[1]] = 1.0
            
        # Mark Opponent (if enabled)
        if self.enable_opponent:
            opp_pos = self.opponent.pos
            
            grid_map[opp_pos[0], opp_pos[1]] = -2.0
            
        obs_list.extend(grid_map.flatten().tolist())
            
        return np.array(obs_list, dtype=np.float32)

    def _estimate_optimal_steps(self):
        """Rough estimate of optimal steps for this episode"""
        # Trips needed = ceil(total_cargos / max_carry)
        trips = (self.total_cargos + self.max_carry - 1) // self.max_carry
        # Each trip: avg distance to cargo + back
        avg_dist = (self.grid_rows + self.grid_cols) // 2
        return trips * avg_dist * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.robot.reset()
        
        # Generate obstacles using the generator
        safe_zones = [[0, 0]]
        if self.enable_opponent:
            safe_zones.append([self.grid_rows-1, self.grid_cols-1])
        self.obstacles = self.obstacle_generator.generate(safe_zones)
        
        targets = self._spawn_cargos()
        self.robot.set_environment(targets, self.obstacles)
        
        obs = self._get_obs()
        
        # Initialize previous distance for reward shaping
        _, _, self.prev_dist = self._get_nearest_target_info()
        self.last_pos = None  # Track last position to punish oscillation

        if self.render_mode == 'human':
            self.render()

        return obs, {"total_cargos": self.total_cargos}


    def step(self, action, opponent_action=None):
        # 1. Opponent Move (if enabled)
        if self.enable_opponent:
            if opponent_action is not None:
                # External control
                try: 
                    # Handle raw integer input from Gym/SB3
                    op_act = wr.RobotAction(opponent_action)
                except ValueError:
                    op_act = wr.RobotAction(int(opponent_action))
                    
                self.opponent.move(op_act)
            else:
                # Internal heuristic or model
                opp_action = self._get_opponent_action()
                self.opponent.move(opp_action)
        
        # 2. Agent Move
        result = self.agent.move(wr.RobotAction(action))
        
        my_delivered = self.agent.delivered_count
        opp_delivered = self.opponent.delivered_count if self.enable_opponent else 0
        
        # 3. Calculate distance reward shaping and state
        _, _, curr_dist = self._get_nearest_target_info()
        
        # Check collision with rival
        hit_rival = False
        if self.enable_opponent:
            hit_rival = (self.robot.robot_pos == self.robot.robot2_pos)
        
        # Check if all cargos delivered
        targets_empty = not self.robot.targets
        bot1_empty = (self.robot.carrying == 0)
        bot2_empty = (self.robot.robot2_carrying == 0)
        
        all_cleared = targets_empty and bot1_empty and bot2_empty
        
        # Build state dict for reward strategy
        reward_state = {
            "prev_dist": self.prev_dist,
            "curr_dist": curr_dist,
            "is_backtrack": self.last_pos is not None and np.array_equal(self._get_obs()[:2], self.last_pos[:2]), 
            "hit_rival": hit_rival,
            "all_cleared": all_cleared,
            "optimal_steps": self._estimate_optimal_steps() if all_cleared else 0,
            "actual_steps": self.robot.step_count, 
            "my_delivered": my_delivered,
            "opponent_delivered": opp_delivered,
        }
        
        # Calculate reward using polymorphic strategy
        reward = self.reward_strategy.calculate(result, reward_state)
        
        self.prev_dist = curr_dist
        self.last_pos = self._get_obs()[:2]
        
        # Termination conditions
        terminated = False
        if all_cleared:
            info = {
                "task_completed": True,
                "total_cargos": self.total_cargos, 
                "delivered": self.agent.delivered_count
            }
            return self._get_obs(), reward, True, False, info
            
        if self.rewards.get('terminate_on_collision') and (result['hit_obstacle'] or hit_rival):
            info = {
                "task_completed": False,
                "total_cargos": self.total_cargos, 
                "delivered": self.agent.delivered_count
            }
            return self._get_obs(), reward, True, False, info
            
        # Max steps handling usually done by Wrapper, but we have internal check too if needed
        # We leave truncation to Gym wrapper or main loop check
        
        info = {
            "task_completed": False, 
            "total_cargos": self.total_cargos,
            "delivered": self.agent.delivered_count
        }
        
        # Check timeout
        truncated = self.robot.step_count >= self.max_steps
        
        # Clamp reward - Optional
        # reward = max(-1.0, min(1.0, reward))
            
        obs = self._get_obs()
        dx, dy, dist = self._get_nearest_target_info() # Recalc for info if needed, but we have curr_dist

        info = {
            "carrying": result["carrying"],
            "delivered": my_delivered,
            "total_cargos": self.total_cargos,
            "remaining": len(self.robot.targets),
            "steps": self.robot.step_count,
            "picked_cargo": result["picked_cargo"],
            "just_delivered": result["delivered"],
            "task_completed": all_cleared,
            "timeout": truncated
        }

        if self.render_mode == 'human':
            self.render()

        return obs, reward * self.reward_scaler, terminated, truncated, info

    def render(self):
        remaining = len(self.robot.targets)
        trips_done = self.robot.delivered_count // self.max_carry if self.robot.delivered_count > 0 else 0
        
        if not self.robot.can_pick_more() or (remaining == 0 and self.robot.carrying > 0):
            phase = f"Return ({self.robot.carrying} cargo)"
        elif remaining > 0:
            phase = f"Collect ({remaining} left)"
        else:
            phase = "Done!"
            
        status = f'{phase} | Delivered: {self.robot.delivered_count}/{self.total_cargos} | Steps: {self.robot.step_count}'
        self.robot.render(status)

if __name__=="__main__":
    env = gym.make('warehouse-robot-v0', render_mode='human')
    
    for ep in range(3):
        obs, info = env.reset()
        print(f"\n=== Episode {ep+1}: {info['total_cargos']} cargos ===")
        
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
            
        print(f"Completed in {info['steps']} steps, Reward: {reward:.2f}")
