# Author: Taylor Tomblin, Michaela Luong
# Login: ttomblin, michalu
# Date: 2/28/2025
# Description: This program implements a Robot Localization system using the Hidden Markov Model (HMM) update step based algorithm.

import numpy as np

"""
Please make sure to install numpy before running this code
To install numpy, run the following command your the command prompt:  pip3 install numpy
"""

# Define the maze layout
# 0: open square, 1: obstacle
maze = np.array([
  [1, 1, 0, 0, 1, 1, 0, 0, 1],
  [1, 1, 1, 0, 1, 1, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 0, 1, 1, 0, 1, 1]
])

# Define the possible movements and their probabilities
movement_probs = {
  'N': {'straight': 0.7, 'left': 0.1, 'right': 0.2},
  'S': {'straight': 0.7, 'left': 0.1, 'right': 0.2},
  'E': {'straight': 0.7, 'left': 0.1, 'right': 0.2},
  'W': {'straight': 0.7, 'left': 0.1, 'right': 0.2}
}

# Define the initial belief state
# Initialize the belief state with equal probability for all open squares
belief = np.full(maze.shape, 1.0 / np.sum(maze == 0))

# Function to normalize the belief state
def normalize(belief):
  return belief / np.sum(belief)

# Function to update the belief state using the HMM algorithm
def hmm_update(belief, action, observation):
  new_belief = np.zeros_like(belief)
  for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
      if maze[i, j] == 0:
        for move, prob in movement_probs[action].items():
          ni, nj = i, j
          if move == 'straight':
            ni, nj = (i - 1, j) if action == 'N' else (i + 1, j) if action == 'S' else (i, j + 1) if action == 'E' else (i, j - 1)
          elif move == 'left':
            ni, nj = (i, j - 1) if action == 'N' else (i, j + 1) if action == 'S' else (i - 1, j) if action == 'E' else (i + 1, j)
          elif move == 'right':
            ni, nj = (i, j + 1) if action == 'N' else (i, j - 1) if action == 'S' else (i + 1, j) if action == 'E' else (i - 1, j)
          
          # Check if the new position is within bounds and not an obstacle
          if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj] == 0:
            new_belief[ni, nj] += belief[i, j] * prob
          else:
            # If the move results in hitting an obstacle, stay in the same position
            new_belief[i, j] += belief[i, j] * prob

  # Normalize the new belief state
  new_belief = normalize(new_belief)
  
  # Update the belief state based on the observation
  for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
      if maze[i, j] == 0:
        new_belief[i, j] *= observation[i, j]
  
  # Normalize the updated belief state
  return normalize(new_belief)

# Function to simulate the robot's sensing action
def sense(maze, position):
  i, j = position
  observation = np.ones(maze.shape)
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  for di, dj in directions:
    ni, nj = i + di, j + dj
    if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1]:
      if maze[ni, nj] == 1:
        observation[i, j] *= 0.95
      else:
        observation[i, j] *= 0.15
  return observation

# Example usage
if __name__ == "__main__":
  # Define an example initial position
  initial_position = (2, 2)
  
  # Perform an HMM update with action 'N' and the given observation
  observation = sense(maze, initial_position)
  belief = hmm_update(belief, 'N', observation)
  print("Updated belief state:")
  print(belief)