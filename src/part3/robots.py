'''
Robot Agents for Warehouse Environment
This provides different robot Types with polymorphic behavior.
'''
from abc import ABC, abstractmethod
import random
from enum import Enum


class RobotAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class BaseRobot(ABC):
    """
    Abstract base class for all robot agents.
    Handles position, cargo carrying, and movement.
    Subclasses implement different decision-making strategies.
    """
    def __init__(self, start_pos: list, grid_rows: int, grid_cols: int, max_carry: int = 3):
        self.start_pos = list(start_pos)
        self.pos = list(start_pos)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_carry = max_carry
        self.carrying = 0
        self.delivered = 0
    
    def reset(self):
        """Reset robot to starting position"""
        self.pos = list(self.start_pos)
        self.carrying = 0
        self.delivered = 0
    
    def move(self, action: RobotAction, obstacles: list) -> dict:
        """
        Execute movement action.
        Returns dict with result info.
        """
        new_pos = self.pos.copy()
        hit_boundary = False
        
        if action == RobotAction.LEFT:
            if self.pos[1] > 0:
                new_pos[1] -= 1
            else:
                hit_boundary = True
        elif action == RobotAction.RIGHT:
            if self.pos[1] < self.grid_cols - 1:
                new_pos[1] += 1
            else:
                hit_boundary = True
        elif action == RobotAction.UP:
            if self.pos[0] > 0:
                new_pos[0] -= 1
            else:
                hit_boundary = True
        elif action == RobotAction.DOWN:
            if self.pos[0] < self.grid_rows - 1:
                new_pos[0] += 1
            else:
                hit_boundary = True
        
        hit_obstacle = new_pos in obstacles
        
        if not hit_obstacle and not hit_boundary:
            self.pos = new_pos
        
        return {
            "hit_obstacle": hit_obstacle or hit_boundary,
            "moved": not (hit_obstacle or hit_boundary),
            "position": self.pos.copy()
        }
    
    def pickup(self, targets: list) -> bool:
        """Try to pick up cargo at current position"""
        if self.pos in targets and self.carrying < self.max_carry:
            targets.remove(self.pos)
            self.carrying += 1
            return True
        return False
    
    def deliver(self) -> int:
        """Deliver cargo at home position"""
        if self.pos == self.start_pos and self.carrying > 0:
            count = self.carrying
            self.delivered += count
            self.carrying = 0
            return count
        return 0
    
    def can_carry_more(self) -> bool:
        return self.carrying < self.max_carry
    
    @abstractmethod
    def get_action(self, targets: list, obstacles: list) -> RobotAction:
        """
        Decide next action based on environment state.
        Implemented by subclasses.
        """
        pass


class GreedyRobot(BaseRobot):
    """
    Robot that moves toward nearest target using greedy strategy.
    If full or no targets, returns to home base.
    """
    def get_action(self, targets: list, obstacles: list) -> RobotAction:
        # If full or no targets, go home
        if self.carrying >= self.max_carry or not targets:
            dest = self.start_pos
        else:
            # Find nearest target
            min_dist = float('inf')
            dest = targets[0] if targets else self.start_pos
            for t in targets:
                dist = abs(t[0] - self.pos[0]) + abs(t[1] - self.pos[1])
                if dist < min_dist:
                    min_dist = dist
                    dest = t
        
        # Move toward destination
        dy = dest[0] - self.pos[0]
        dx = dest[1] - self.pos[1]
        
        if abs(dy) > abs(dx):
            return RobotAction.DOWN if dy > 0 else RobotAction.UP
        else:
            return RobotAction.RIGHT if dx > 0 else RobotAction.LEFT


class RandomRobot(BaseRobot):
    """
    Robot that moves randomly.
    Useful for testing or as a baseline.
    """
    def get_action(self, targets: list, obstacles: list) -> RobotAction:
        return random.choice(list(RobotAction))


class PatrolRobot(BaseRobot):
    """
    Robot that patrols in a fixed pattern.
    Cycles through: RIGHT -> DOWN -> LEFT -> UP
    """
    def __init__(self, start_pos: list, grid_rows: int, grid_cols: int, max_carry: int = 3):
        super().__init__(start_pos, grid_rows, grid_cols, max_carry)
        self.patrol_index = 0
        self.patrol_pattern = [
            RobotAction.RIGHT, RobotAction.RIGHT,
            RobotAction.DOWN, RobotAction.DOWN,
            RobotAction.LEFT, RobotAction.LEFT,
            RobotAction.UP, RobotAction.UP
        ]
    
    def get_action(self, targets: list, obstacles: list) -> RobotAction:
        action = self.patrol_pattern[self.patrol_index]
        self.patrol_index = (self.patrol_index + 1) % len(self.patrol_pattern)
        return action


def create_robot(robot_type: str, start_pos: list, grid_rows: int, grid_cols: int, 
                 max_carry: int = 3) -> BaseRobot:
    """Factory function to create robots by type name."""
    robots = {
        'greedy': GreedyRobot,
        'random': RandomRobot,
        'patrol': PatrolRobot,
    }
    
    robot_class = robots.get(robot_type.lower())
    if robot_class is None:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    return robot_class(start_pos, grid_rows, grid_cols, max_carry)


if __name__ == "__main__":
    print("=== GreedyRobot ===")
    greedy = GreedyRobot([0, 0], 8, 8)
    targets = [[3, 3], [5, 5]]
    action = greedy.get_action(targets, [])
    print(f"At {greedy.pos}, targets={targets}, action={action}")
    
    print("\n=== RandomRobot ===")
    rand_bot = RandomRobot([0, 0], 8, 8)
    for i in range(3):
        action = rand_bot.get_action([], [])
        print(f"Random action {i+1}: {action}")
    
    print("\n=== PatrolRobot ===")
    patrol = PatrolRobot([4, 4], 8, 8)
    for i in range(4):
        action = patrol.get_action([], [])
        print(f"Patrol step {i+1}: {action}")
    
    print("\n=== Factory ===")
    bot = create_robot('greedy', [7, 7], 8, 8)
    print(f"Created: {type(bot).__name__} at {bot.pos}")
