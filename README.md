# OOP Project - Reinforcement Learning

This project explores Reinforcement Learning (RL) concepts using Gymnasium environments. It consists of three parts, covering Q-Learning, SARSA, and a custom environment implementation.

## Project Structure

- **part1/**: Q-Learning implementation for the Mountain Car environment.
- **part2/**: SARSA implementation for the Frozen Lake (8x8) environment.
- **part3/**: Custom Gymnasium environment simulation for a Warehouse Robot.

## Prerequisites

Ensure you have Python 3.13+ installed. This project uses `uv` for dependency management, but you can also use `pip`.

### Dependencies
- `gymnasium`
- `numpy`
- `matplotlib`
- `pygame` (required for Part 3 and rendering)
- `ipykernel` (for notebook execution)

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    If you are using `uv`:
    ```bash
    uv sync
    ```
    Or with `pip`:
    ```bash
    pip install gymnasium numpy matplotlib pygame ipykernel
    ```

## Usage

### Part 1: Mountain Car (Q-Learning)

Solves the classic Mountain Car problem where an underpowered car must drive up a steep hill.

**Training:**
To train the agent and save the Q-table:
```bash
python part1/mountain_car.py --train --episodes 10000
```
This will save the model to `part1/mountain_car.pkl` and a reward plot to `part1/mountain_car.png`.

**Evaluation:**
To run the trained agent with rendering:
```bash
python part1/mountain_car.py --episodes 10 --render
```

### Part 2: Frozen Lake (SARSA)

Solves the Frozen Lake 8x8 environment where an agent must navigate a frozen lake from Start to Goal without falling into holes.

**Training:**
To train the agent:
```bash
python part2/frozen_lake.py --train --episodes 10000
```
This saves the model to `part2/frozen_lake8x8.pkl` and a reward plot to `part2/frozen_lake8x8.png`.

**Evaluation:**
To evaluate the trained agent:
```bash
python part2/frozen_lake.py --episodes 100 --render
```

### Part 3: Warehouse Robot (Custom Environment)

A custom GridWorld environment where a robot must navigate to a target package.

**Running the Environment:**
You can run the environment directly to see the robot in action (random actions or manual control if implemented):
```bash
python part3/oop_project_env.py
```
Or run the robot logic directly:
```bash
python part3/warehouse_robot.py
```

## Implementation Details

-   **Part 1**: Uses Q-Learning with state discretization for continuous observation space.
-   **Part 2**: Uses SARSA (State-Action-Reward-State-Action) algorithm.
-   **Part 3**: Implements a custom class `WarehouseRobot` and wraps it in a Gymnasium `Env` class (`WarehouseRobotEnv`). Uses Pygame for rendering.
