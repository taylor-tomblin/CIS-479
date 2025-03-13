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
belief = np.full(maze.shape, 1.0 / np.sum(maze == 0))

def normalize(belief):
    return belief / np.sum(belief)

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
                    
                    if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj] == 0:
                        new_belief[ni, nj] += belief[i, j] * prob
                    else:
                        new_belief[i, j] += belief[i, j] * prob

    new_belief = normalize(new_belief)
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0:
                new_belief[i, j] *= observation[i, j]
    
    return normalize(new_belief)

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

# Define the sensing probabilities
sensing_probs = {
  'obstacle': {'detected': 0.95, 'not_detected': 0.05},
  'open': {'detected': 0.15, 'not_detected': 0.85}
}

# Initialize the probability distribution
def initialize_probabilities(maze):
  open_squares = np.where(maze == 0)
  num_open_squares = len(open_squares[0])
  prob = np.zeros_like(maze, dtype=float)
  prob[open_squares] = 1.0 / num_open_squares
  return prob

# Update probabilities based on sensing evidence
def update_sensing(prob, evidence):
  for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
      if maze[i, j] == 0:
        # Calculate the probability of observing the evidence given the state
        obs_prob = 1.0
        for k, direction in enumerate(['W', 'N', 'E', 'S']):
          if evidence[k] == 1:
            obs_prob *= sensing_probs['obstacle']['detected'] if is_obstacle(i, j, direction) else sensing_probs['open']['detected']
          else:
            obs_prob *= sensing_probs['obstacle']['not_detected'] if is_obstacle(i, j, direction) else sensing_probs['open']['not_detected']
        prob[i, j] *= obs_prob
  # Normalize the probabilities
  prob /= np.sum(prob)
  return prob

# Check if there is an obstacle in a given direction
def is_obstacle(i, j, direction):
  if direction == 'W':
    return j == 0 or maze[i, j-1] == 1
  elif direction == 'N':
    return i == 0 or maze[i-1, j] == 1
  elif direction == 'E':
    return j == maze.shape[1] - 1 or maze[i, j+1] == 1
  elif direction == 'S':
    return i == maze.shape[0] - 1 or maze[i+1, j] == 1
  return False

# Predict the next state probabilities based on movement
def predict_movement(prob, action):
  new_prob = np.zeros_like(prob)
  for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
      if maze[i, j] == 0:
        # Calculate the new position probabilities
        for move, move_prob in movement_probs[action].items():
          new_i, new_j = get_new_position(i, j, action, move)
          if maze[new_i, new_j] == 0:
            new_prob[new_i, new_j] += prob[i, j] * move_prob
          else:
            new_prob[i, j] += prob[i, j] * move_prob
  # Normalize the probabilities
  new_prob /= np.sum(new_prob)
  return new_prob

# Get the new position after a movement
def get_new_position(i, j, action, move):
  if move == 'straight':
    if action == 'N':
      return max(i-1, 0), j
    elif action == 'S':
      return min(i+1, maze.shape[0]-1), j
    elif action == 'W':
      return i, max(j-1, 0)
    elif action == 'E':
      return i, min(j+1, maze.shape[1]-1)
  elif move == 'left':
    if action == 'N':
      return i, max(j-1, 0)
    elif action == 'S':
      return i, min(j+1, maze.shape[1]-1)
    elif action == 'W':
      return min(i+1, maze.shape[0]-1), j
    elif action == 'E':
      return max(i-1, 0), j
  elif move == 'right':
    if action == 'N':
      return i, min(j+1, maze.shape[1]-1)
    elif action == 'S':
      return i, max(j-1, 0)
    elif action == 'W':
      return max(i-1, 0), j
    elif action == 'E':
      return min(i+1, maze.shape[0]-1), j
  return i, j

# Helper function to format probabilities
def format_probabilities(prob):
  formatted_prob = np.zeros_like(prob, dtype=object)
  for i in range(prob.shape[0]):
    for j in range(prob.shape[1]):
      if prob[i, j] < 0.00000001:
        formatted_prob[i, j] = "########"
      else:
        formatted_prob[i, j] = f"{prob[i, j] * 100:.2f}"
  return formatted_prob

# Main function to simulate the robot's actions
def main():
  # Initialize probabilities
  prob = initialize_probabilities(maze)
  print("Initial Location Probabilities")
  print(format_probabilities(prob))

  # Sequence of actions
  actions = [
    ('sensing', [1, 0, 1, 1]),
    ('moving', 'N'),
    ('sensing', [0, 0, 1, 1]),
    ('moving', 'E'),
    ('sensing', [0, 0, 0, 1]),
    ('moving', 'N'),
    ('sensing', [0, 1, 0, 0]),
    ('moving', 'W'),
    ('sensing', [0, 0, 0, 0]),
    ('moving', 'S'),
    ('sensing', [0, 0, 0, 0])
  ]

  for action_type, action_value in actions:
    if action_type == 'sensing':
      prob = update_sensing(prob, action_value)
      print(f"\nFiltering after Evidence {action_value}")
      print(format_probabilities(prob))
    elif action_type == 'moving':
      prob = predict_movement(prob, action_value)
      print(f"\nPrediction after Action {action_value}")
      print(format_probabilities(prob))

if __name__ == "__main__":
  main()
