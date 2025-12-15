'''
Reward Computation Strategies for Warehouse Robot Environment
Different reward strategies for training agents with various objectives.
'''
from abc import ABC, abstractmethod
class RewardStrategy(ABC):
    """
    Base class for reward calculation strategies.
    Subclasses define different reward shaping approaches.
    """
    def __init__(self, config: dict = None):
        # Default reward parameters
        self.config = config if config else {
            "step_penalty": -0.02,
            "collision_penalty": -1.0,
            "pickup_reward": 0.5,
            "delivery_base": 1.0,
            "delivery_combo": 0.5,
            "shaping_factor": 0.05,
            "efficiency_bonus": 0.2
        }
    
    @abstractmethod
    def calculate(self, result: dict, state: dict) -> float:
        """
        Calculate reward based on action result and environment state.
        
        Args:
            result: Action result from robot.move() containing:
                - hit_obstacle, moved, picked_cargo, delivered, etc.
            state: Current environment state containing:
                - prev_dist, curr_dist, last_pos, current_pos, etc.
        
        Returns:
            Calculated reward value
        """
        pass
    
    def get(self, key: str, default=None):
        """Get config value with fallback"""
        return self.config.get(key, default)


class BasicReward(RewardStrategy):
    """
    Simple reward: step penalty + pickup/delivery rewards only.
    No distance shaping - good for baseline comparison.
    """
    def calculate(self, result: dict, state: dict) -> float:
        reward = self.get('step_penalty', -0.02)
        
        # Collision penalty
        if result.get('hit_obstacle') or state.get('hit_rival'):
            return self.get('collision_penalty', -1.0)
        
        # Pickup reward
        if result.get('picked_cargo'):
            reward += self.get('pickup_reward', 0.5)
        
        # Delivery reward
        delivered = result.get('delivered', 0)
        if delivered > 0:
            reward += self.get('delivery_base', 1.0) * delivered
        
        return reward


class ShapingReward(RewardStrategy):
    """
    Distance-based reward shaping.
    Encourages agent to move closer to targets - helps with exploration.
    """
    def calculate(self, result: dict, state: dict) -> float:
        # Base step penalty
        reward = self.get('step_penalty', -0.02)
        
        # Distance shaping: reward for getting closer
        prev_dist = state.get('prev_dist', 0)
        curr_dist = state.get('curr_dist', 0)
        shaping = (prev_dist - curr_dist) * self.get('shaping_factor', 0.05)
        reward += shaping
        
        # Oscillation penalty
        if state.get('is_backtrack'):
            reward -= 0.1
        
        # Collision
        if result.get('hit_obstacle') or state.get('hit_rival'):
            return self.get('collision_penalty', -1.0)
        
        # Pickup
        if result.get('picked_cargo'):
            reward += self.get('pickup_reward', 0.5)
        
        # Delivery with combo bonus
        delivered = result.get('delivered', 0)
        if delivered > 0:
            base = self.get('delivery_base', 1.0)
            combo = self.get('delivery_combo', 0.5)
            reward += base * delivered
            if delivered > 1:
                reward += combo * (delivered * (delivered - 1))
        
        # Efficiency bonus on completion
        if state.get('all_cleared'):
            optimal = state.get('optimal_steps', 1)
            actual = state.get('actual_steps', 1)
            eff_scale = self.get('efficiency_bonus', 0.2)
            reward += eff_scale * min(1.0, optimal / max(actual, 1))
        
        return reward


class CompetitiveReward(ShapingReward):
    """
    Extends ShapingReward with competition bonuses.
    Rewards beating the opponent in cargo collection.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        # Additional competitive params
        self.lead_bonus = 0.3
        self.steal_bonus = 0.5
    
    def calculate(self, result: dict, state: dict) -> float:
        # Get base reward from parent
        reward = super().calculate(result, state)
        
        # Bonus for being ahead of opponent
        my_delivered = state.get('my_delivered', 0)
        opp_delivered = state.get('opponent_delivered', 0)
        
        if my_delivered > opp_delivered:
            reward += self.lead_bonus * (my_delivered - opp_delivered)
        
        # Bonus for picking up cargo before opponent reaches it
        if result.get('picked_cargo') and state.get('opponent_was_close'):
            reward += self.steal_bonus
        
        return reward


def create_reward_strategy(strategy_type: str, config: dict = None) -> RewardStrategy:
    """
    Factory function to create reward strategy by name.
    
    Args:
        strategy_type: 'basic', 'shaping', or 'competitive'
        config: Reward configuration dict
    """
    strategies = {
        'basic': BasicReward,
        'shaping': ShapingReward,
        'competitive': CompetitiveReward,
    }
    
    strategy_class = strategies.get(strategy_type.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy_class(config)


if __name__ == "__main__":
    print("=== BasicReward ===")
    basic = BasicReward()
    result = {"picked_cargo": True, "delivered": 0, "hit_obstacle": False}
    state = {}
    print(f"Pickup reward: {basic.calculate(result, state)}")
    
    print("\n=== ShapingReward ===")
    shaping = ShapingReward()
    state = {"prev_dist": 5, "curr_dist": 3, "is_backtrack": False}
    result = {"picked_cargo": False, "delivered": 0, "hit_obstacle": False}
    print(f"Moving closer: {shaping.calculate(result, state)}")
    
    print("\n=== CompetitiveReward ===")
    competitive = CompetitiveReward()
    state = {"prev_dist": 5, "curr_dist": 3, "my_delivered": 3, "opponent_delivered": 1}
    print(f"Leading opponent: {competitive.calculate(result, state)}")
    
    print("\n=== Factory ===")
    strat = create_reward_strategy('shaping')
    print(f"Created: {type(strat).__name__}")
