import gymnasium as gym
import sys
import os
import numpy as np

# Add src/part3 to path so we can import warehouse_env
# sys.path.append(os.path.abspath("src/part3")) # We are inside src/part3

import warehouse_env
import warehouse_robot

def test_stage_1():
    print("\n=== Testing Stage 1: Navigation Only ===")
    env = warehouse_env.WarehouseRobotEnv(stage=1)
    
    # Check config overrides
    assert env.enable_opponent == False, "Stage 1 should have no opponent"
    assert env.max_carry == 1, "Stage 1 should have max_carry=1"
    assert env.max_cargos == 1, "Stage 1 should have max_cargos=1"
    
    obs, info = env.reset()
    target_pos = env.robot.targets[0]
    print(f"Target at: {target_pos}")
    
    # Teleport to target to test immediate termination
    env.robot.robot_pos = list(target_pos)
    
    # Strategy: Teleport to neighbor of target, then move INTO target.
    t_r, t_c = target_pos
    
    # Find a valid neighbor
    start_pos = None
    action_to_take = None
    
    # Try UP (from below)
    if t_r < env.grid_rows - 1:
        start_pos = [t_r + 1, t_c]
        action_to_take = 3 # UP
    # Try DOWN (from above)
    elif t_r > 0:
        start_pos = [t_r - 1, t_c]
        action_to_take = 1 # DOWN
        
    env.robot.robot_pos = list(start_pos)
    print(f"Teleported to {start_pos}, taking action {action_to_take} to hit {target_pos}")
    
    obs, reward, terminated, truncated, info = env.step(action_to_take)
    
    assert info['picked_cargo'] == True, "Should have picked up cargo"
    assert terminated == True, "Stage 1 should terminate immediately on pickup"
    print("Stage 1 Passed: Immediate termination on pickup verified.")
    env.close()

def test_stage_2():
    print("\n=== Testing Stage 2: Cargo Delivery (No Enemy) ===")
    # Explicitly ask for opponent=True in args, but Stage 2 should override it to False
    env = warehouse_env.WarehouseRobotEnv(stage=2, enable_opponent=True, max_carry=3)
    
    assert env.enable_opponent == False, "Stage 2 should disable opponent even if requested"
    assert env.max_carry == 3, "Stage 2 should respect max_carry param"
    
    obs, info = env.reset()
    
    # Verify no opponent in grid (value -2)
    # Obs structure: [r, c, carry, dx, dy, grid...]
    grid_start_idx = 5
    grid_vals = obs[grid_start_idx:]
    assert -2.0 not in grid_vals, "Stage 2 should not have opponent in observation"
    
    # Teleport to target and pickup
    if not env.robot.targets:
        print("Skipping pickup test (no targets generated)")
        return

    target_pos = env.robot.targets[0]
    
    # Neighbor teleport trick again
    t_r, t_c = target_pos
    if t_r < env.grid_rows - 1:
        start_pos = [t_r + 1, t_c]
        action_to_take = 3 # UP
    else:
        start_pos = [t_r - 1, t_c]
        action_to_take = 1 # DOWN

    env.robot.robot_pos = list(start_pos)
    obs, reward, terminated, _, info = env.step(action_to_take)
    
    assert info['picked_cargo'] == True
    assert terminated == False, "Stage 2 should NOT terminate on pickup"
    assert env.robot.carrying > 0
    print("Stage 2 Passed: Pickup continued, no opponent found.")
    env.close()

def test_stage_3():
    print("\n=== Testing Stage 3: Competition ===")
    env = warehouse_env.WarehouseRobotEnv(stage=3, enable_opponent=True)
    
    assert env.enable_opponent == True, "Stage 3 should have opponent"
    
    obs, info = env.reset()
    
    # Verify opponent in grid
    grid_vals = obs[5:]
    assert -2.0 in grid_vals, "Stage 3 MUST have opponent (-2.0) in observation"
    
    print("Stage 3 Passed: Opponent present.")
    env.close()

if __name__ == "__main__":
    test_stage_1()
    test_stage_2()
    test_stage_3()
    print("\nALL STAGE TESTS PASSED!")
