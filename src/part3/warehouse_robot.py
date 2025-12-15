'''
Warehouse Robot - Only handles robot state and rendering
Environment configuration (obstacles, cargos) is handled by WarehouseRobotEnv
'''
import random
from enum import Enum
import pygame
import sys
from os import path

class RobotAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3

class GridTile(Enum):
    _FLOOR=0
    ROBOT=1
    TARGET=2
    OBSTACLE=3

    def __str__(self):
        return self.name[:1]

class WarehouseRobot:
    """
    Pure robot class - only handles:
    - Robot position and movement
    - Carrying cargo (with capacity limit)
    - Rendering the grid
    """
    
    def __init__(self, grid_rows=8, grid_cols=8, fps=1, max_carry=3, enable_opponent=True):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fps = fps
        self.max_carry = max_carry  # Maximum cargo robot can carry at once
        self.enable_opponent = enable_opponent
        self.last_action = ''
        self.collision_active = False  # Track if robot just hit something
        
        # Robot 1 state (Agent)
        self.robot_pos = [0, 0]
        self.carrying = 0
        self.delivered_count = 0
        
        # Robot 2 state (Rival/Partner)
        self.robot2_pos = [self.grid_rows-1, self.grid_cols-1]  # Start at bottom-right
        self.robot2_start = [self.grid_rows-1, self.grid_cols-1]
        self.robot2_carrying = 0
        self.robot2_delivered = 0
        
        self.step_count = 0 
        self.task_completed = False
        
        # Environment elements (set by Env)
        self.targets = []
        self.obstacles = []
        
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()

        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        self.window_size = (self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height)
        self.window_surface = pygame.display.set_mode(self.window_size) 

        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/package.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size) 

    def reset(self):
        """Reset robot to starting position"""
        self.robot_pos = [0, 0]
        self.carrying = 0
        self.delivered_count = 0
        
        # Reset Robot 2
        self.robot2_pos = list(self.robot2_start)
        self.robot2_carrying = 0
        self.robot2_delivered = 0
        
        self.step_count = 0
        self.task_completed = False
        self.targets = []
        self.obstacles = []
        self.obstacles = []
        self.last_action = ''
        self.collision_active = False

    def set_environment(self, targets: list, obstacles: list):
        """Called by Env to set up the environment elements"""
        self.targets = targets.copy()
        self.obstacles = obstacles.copy()

    def move(self, robot_action: RobotAction) -> dict:
        """
        Execute a movement action.
        Returns dict with movement result info.
        """
        self.last_action = robot_action
        self.step_count += 1

        # Calculate new position
        new_pos = self.robot_pos.copy()
        hit_boundary = False
        
        if robot_action == RobotAction.LEFT:
            if self.robot_pos[1] > 0:
                new_pos[1] -= 1
            else:
                hit_boundary = True
        elif robot_action == RobotAction.RIGHT:
            if self.robot_pos[1] < self.grid_cols - 1:
                new_pos[1] += 1
            else:
                hit_boundary = True
        elif robot_action == RobotAction.UP:
            if self.robot_pos[0] > 0:
                new_pos[0] -= 1
            else:
                hit_boundary = True
        elif robot_action == RobotAction.DOWN:
            if self.robot_pos[0] < self.grid_rows - 1:
                new_pos[0] += 1
            else:
                hit_boundary = True
        
        # Check obstacle collision
        hit_obstacle = new_pos in self.obstacles
        
        # Move only if no collision
        if not hit_obstacle and not hit_boundary:
            self.robot_pos = new_pos
        
        self.collision_active = hit_obstacle or hit_boundary

        # Check cargo pickup (only if not at max capacity)
        picked_cargo = False
        if self.robot_pos in self.targets and self.carrying < self.max_carry:
            self.targets.remove(self.robot_pos)
            self.carrying += 1
            picked_cargo = True
        
        # Check delivery at origin
        delivered = 0
        if self.robot_pos == [0, 0] and self.carrying > 0:
            delivered = self.carrying
            self.delivered_count += self.carrying
            self.carrying = 0
        
        return {
            "hit_obstacle": hit_obstacle or hit_boundary,  # Threat both as collision
            "moved": not (hit_obstacle or hit_boundary),
            "picked_cargo": picked_cargo,
            "delivered": delivered,
            "at_origin": self.robot_pos == [0, 0],
            "position": self.robot_pos.copy(),
            "carrying": self.carrying,
            "total_delivered": self.delivered_count
        }

    def move_bot2(self, action: RobotAction):
        """Move Robot 2 (The Rival/Partner)"""
        new_pos = self.robot2_pos.copy()
        
        if action == RobotAction.LEFT and self.robot2_pos[1] > 0:
            new_pos[1] -= 1
        elif action == RobotAction.RIGHT and self.robot2_pos[1] < self.grid_cols - 1:
            new_pos[1] += 1
        elif action == RobotAction.UP and self.robot2_pos[0] > 0:
            new_pos[0] -= 1
        elif action == RobotAction.DOWN and self.robot2_pos[0] < self.grid_rows - 1:
            new_pos[0] += 1
            
        # Bot 2 checks obstacles too
        if new_pos not in self.obstacles:
            self.robot2_pos = new_pos
            
        # Bot 2 Pickup (competes for same targets!)
        picked = False
        if self.robot2_pos in self.targets and self.robot2_carrying < self.max_carry:
            self.targets.remove(self.robot2_pos)
            self.robot2_carrying += 1
            picked = True
            
        # Bot 2 Delivery (at its own origin [7,7])
        if self.robot2_pos == self.robot2_start and self.robot2_carrying > 0:
            self.robot2_delivered += self.robot2_carrying
            self.robot2_carrying = 0
            
        return picked

    def get_position(self) -> list:
        return self.robot_pos.copy()

    def can_pick_more(self) -> bool:
        return self.carrying < self.max_carry

    def render(self, status_text: str = ""):
        """Render the grid with current state"""
        # Console output
        # for r in range(self.grid_rows):
        #     for c in range(self.grid_cols):
        #         if [r, c] == self.robot_pos:
        #             print(GridTile.ROBOT, end=' ')
        #         elif [r, c] in self.targets:
        #             print(GridTile.TARGET, end=' ')
        #         elif [r, c] in self.obstacles:
        #             print(GridTile.OBSTACLE, end=' ')
        #         else:
        #             print(GridTile._FLOOR, end=' ')
        #     print()
        # print()

        self._process_events()
        self.window_surface.fill((255, 255, 255))

        # Pygame render
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                
                # Draw floor
                pygame.draw.rect(self.window_surface, (240, 240, 240), 
                               (pos[0], pos[1], self.cell_width, self.cell_height), 1)

                # Draw Origin 1 (Green)
                if [r, c] == [0, 0]:
                    origin_rect = pygame.Rect(pos[0], pos[1], self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (100, 200, 100), origin_rect, 4)

                # Draw Origin 2 (Blue) - Bot 2 Base (Only if enabled)
                if self.enable_opponent and [r, c] == self.robot2_start:
                    origin2_rect = pygame.Rect(pos[0], pos[1], self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (100, 100, 200), origin2_rect, 4)

                # Draw obstacles
                if [r, c] in self.obstacles:
                    obstacle_rect = pygame.Rect(pos[0], pos[1], self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (60, 60, 60), obstacle_rect)

                # Draw targets
                if [r, c] in self.targets:
                    self.window_surface.blit(self.goal_img, pos)

                # Draw Robot 1
                if [r, c] == self.robot_pos:
                    self.window_surface.blit(self.robot_img, pos)
                    # Always show carrying count
                    carry_str = str(self.carrying)
                    
                    # Position relative to cell (top-left)
                    text_x, text_y = pos[0] + 5, pos[1] + 5
                    
                    # Color: White usually, Red if full
                    text_color = (0, 0, 0)
                    if self.carrying == self.max_carry:
                        text_color = (255, 150, 150) # Light red when full

                    carry_text = self.action_font.render(carry_str, True, text_color)
                    
                    # Add a black shadow for readability
                    shadow_text = self.action_font.render(carry_str, True, (0, 0, 0))
                    self.window_surface.blit(shadow_text, (text_x , text_y))
                    self.window_surface.blit(carry_text, (text_x, text_y))

                    # Draw red box on collision
                    if self.collision_active:
                         pygame.draw.rect(self.window_surface, (255, 0, 0), 
                                (pos[0], pos[1], self.cell_width, self.cell_height), 4)

                # Draw Robot 2 (Blue)
                if self.enable_opponent and [r, c] == self.robot2_pos:
                    # Draw blue robot
                    pygame.draw.circle(self.window_surface, (0, 0, 255), 
                                     (pos[0] + self.cell_width//2, pos[1] + self.cell_height//2), 
                                     self.cell_width//3)
                    # carry count
                    if self.robot2_carrying > 0:
                        carry2_text = self.action_font.render(str(self.robot2_carrying), True, (255, 255, 255))
                        self.window_surface.blit(carry2_text, (pos[0] + 5, pos[1] + 5))

        # Status bar
        if not status_text:
            if self.enable_opponent:
                status_text = f'P1: {self.delivered_count} | P2: {self.robot2_delivered}'
            else:
                status_text = f'Delivered: {self.delivered_count} | Steps: {self.step_count}'
        
        display_text = f'{self.last_action} | {status_text}'
        text_img = self.action_font.render(display_text, True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
        self.clock.tick(self.fps)  

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

if __name__=="__main__":
    robot = WarehouseRobot(max_carry=3)
    robot.set_environment(
        targets=[[2, 3], [5, 5], [1, 6], [6, 2]],
        obstacles=[[3, 3], [3, 4]]
    )
    robot.render()

    while True:
        rand_action = random.choice(list(RobotAction))
        result = robot.move(rand_action)
        print(f"Carrying: {result['carrying']}, Delivered: {result['total_delivered']}")
        robot.render()
