'''
Obstacle Generation Strategies for Warehouse Environment
This Provides different obstacle placement strategies:
- Fixed maze layout for consistent training
- Random obstacles for generalization
- Empty mode for debugging
'''
import random
from abc import ABC, abstractmethod


class ObstacleGenerator(ABC):
    """
    Base class for obstacle generation strategies.
    Subclasses implement different placement algorithms.
    """
    def __init__(self, grid_rows: int, grid_cols: int):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    @abstractmethod
    def generate(self, safe_zones: list = None) -> list:
        """
        Generate obstacle positions for the grid.
        
        Args:
            safe_zones: List of [row, col] positions to avoid
            
        Returns:
            List of [row, col] obstacle positions
        """
        pass
    
    def _filter_safe_zones(self, obstacles: list, safe_zones: list) -> list:
        """Remove any obstacles that overlap with safe zones"""
        if not safe_zones:
            return obstacles
        return [obs for obs in obstacles if obs not in safe_zones]


class FixedObstacleGenerator(ObstacleGenerator):
    """
    Generates a fixed maze-like obstacle layout.
    Good for training where consistent environment is needed.
    
    Layout for 8x8 grid:
    
        0 1 2 3 4 5 6 7
        _ _ _ _ _ _ _ _  0
        _ _ O O O _ _ _  1  (horizontal barrier)
        _ _ O _ _ _ O _  2  (vertical walls)
        _ _ O _ _ _ O _  3
        _ _ _ _ O O O _  4  (middle barrier)
        _ O _ _ _ _ _ _  5  
        _ O O _ _ _ O _  6  (L-shape and corner)
        _ _ _ _ _ _ O _  7
    """
    
    def generate(self, safe_zones: list = None) -> list:
        # Maze layout - forces navigation around walls
        obstacles = [
            # Horizontal barrier top
            [1, 2], [1, 3], [1, 4],
            
            # Vertical wall left
            [2, 2], [3, 2],
            
            # Vertical wall right
            [2, 6], [3, 6],
            
            # Horizontal barrier middle
            [4, 4], [4, 5], [4, 6],
            
            # L-shape bottom left
            [5, 1], [6, 1], [6, 2],
            
            # Corner blocker bottom right
            [6, 6], [7, 6],
        ]
        
        return self._filter_safe_zones(obstacles, safe_zones)


class RandomObstacleGenerator(ObstacleGenerator):
    """
    Generates random obstacles each episode.
    Helps with generalization - agent learns to navigate any layout.
    """
    def __init__(self, grid_rows: int, grid_cols: int, 
                 min_obstacles: int = 10, max_obstacles: int = 20):
        super().__init__(grid_rows, grid_cols)
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
    
    def generate(self, safe_zones: list = None) -> list:
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        obstacles = []
        
        if safe_zones is None:
            safe_zones = []
        
        attempts = 0
        max_attempts = num_obstacles * 10  # Prevent infinite loop
        
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            r = random.randint(0, self.grid_rows - 1)
            c = random.randint(0, self.grid_cols - 1)
            candidate = [r, c]
            
            # Avoid safe zones and duplicates
            if candidate not in safe_zones and candidate not in obstacles:
                obstacles.append(candidate)
            attempts += 1
        
        return obstacles


class EmptyObstacleGenerator(ObstacleGenerator):
    """
    No obstacles - for easy mode or debugging.
    """
    
    def generate(self, safe_zones: list = None) -> list:
        return []


# Factory function for convenience
def create_obstacle_generator(generator_type: str, grid_rows: int, grid_cols: int, 
                               **kwargs) -> ObstacleGenerator:
    """
    Create an obstacle generator by type name.
    
    Args:
        generator_type: 'fixed', 'random', or 'empty'
        grid_rows: Grid height
        grid_cols: Grid width
        **kwargs: Additional args for specific generators
        
    Returns:
        ObstacleGenerator instance
    """
    generators = {
        'fixed': FixedObstacleGenerator,
        'random': RandomObstacleGenerator,
        'empty': EmptyObstacleGenerator,
    }
    
    gen_class = generators.get(generator_type.lower())
    if gen_class is None:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    if generator_type.lower() == 'random':
        return gen_class(grid_rows, grid_cols, **kwargs)
    return gen_class(grid_rows, grid_cols)


if __name__ == "__main__":
    print("=== Fixed Generator ===")
    fixed_gen = FixedObstacleGenerator(8, 8)
    obstacles = fixed_gen.generate(safe_zones=[[0, 0]])
    print(f"Fixed: {len(obstacles)} obstacles")
    
    print("\n=== Random Generator ===")
    random_gen = RandomObstacleGenerator(8, 8, min_obstacles=5, max_obstacles=10)
    for i in range(3):
        obstacles = random_gen.generate(safe_zones=[[0, 0], [7, 7]])
        print(f"Run {i+1}: {len(obstacles)} obstacles")
    
    print("\n=== Empty Generator ===")
    empty_gen = EmptyObstacleGenerator(8, 8)
    print(f"Empty: {len(empty_gen.generate())} obstacles")
    
    print("\n=== Factory Function ===")
    gen = create_obstacle_generator('random', 8, 8, min_obstacles=3, max_obstacles=5)
    print(f"Factory created: {type(gen).__name__}")
