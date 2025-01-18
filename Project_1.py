# Author: Taylor Tomblin & Michael Luong
# Login: ttomblin, michalu
# Date: 1/9/2025
# Description: This program solves the 8-puzzle problem using the A* search algorithm with the Manhattan distance heuristic.
# The program reads the initial and goal states of the puzzle from a file and outputs the solution to the console.
# The program also outputs the number of nodes expanded and the number of nodes in the open list at the end of the search.

import heapq

initial_state = ((1, 6, 2),
                 (5, 7, 8),
                 (0, 4, 3))
                   
goal_state = ((7, 8, 1),
              (6, 0, 2),
              (5, 4, 3))

class Board:
  def __init__(self, board):
    # Initialize the board
    self.board = board
    
  def __eq__(self, other):
    # Check if the states are equal
    return self.state == other.state
    
  def __hash__(self):
    # Return the hash of the state
    return hash(str(self.state))
    
  def __str__(self):
    # Return the string representation of the state
    return str(self.state)
    
  def __repr__(self):
    # Return the string representation of the state
    return str(self.state)

  def get_blank_position(board):
      # Get the position of the blank tile
      for i, row in enumerate(board):
        for j, value in enumerate(row):
          if value == 0:
            return (i, j)

  def swap_positions(board, pos1, pos2):
    # Swap two positions on the board
    board = [list(row) for row in board]
    board[pos1[0]][pos1[1]], board[pos2[0]][pos2[1]] = board[pos2[0]][pos2[1]], board[pos1[0]][pos1[1]]

    return tuple(tuple(row) for row in board)

class PuzzleState:
  def __init__(self, state, parent=None, move=None):
    # Initialize the state
    self.state = state
    self.parent = parent
    self.move = move
    self.g = 0
    self.h = 0
    self.f = 0

  def __lt__(self, other):
    # Compare the states
    return self.f < other.f

def is_goal_state(state):
  # Check if current state matches
  return state.board == goal_state
  
def generate_next_states(current_state):
  # Generate possible next states
    next_states = []
    blank_pos = current_state.board.get_blank_position(current_state.board)
    directions = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
        }
    
    for direction in directions:
      new_blank_pos = (blank_pos[0] + directions[direction][0], blank_pos[1] + directions[direction][1])
        
      if 0 <= new_blank_pos[0] < len(current_state.board) and 0 <= new_blank_pos[1] < len(current_state.board[0]):
        new_board = Board()
        new_board.swap_positions(current_state.board, blank_pos, new_blank_pos)
        next_states.append(PuzzleState(new_board, current_state))
    
    return next_states

def reconstruct_path(state):
    path = []

    while state:
      path.append(state.board)
      state = state.parent

    return path[::-1]

def heuristic(board, goal):
  # Determine the heuristic value
  size = len(board)
  dist = 0
  tiles_out_of_place = 0

  # Find the position of the value in the goal board
  goal_positions = {value: (i, j) for i, row in enumerate(goal) for j, value in enumerate(row)}

  for i in range(size):
    for j in range(size):
      value = board[i][j]
      if value != 0: # skipping the empty tile
        goal_i, goal_j = goal_positions[value] # mapping the value found earlier to goal_i, goal_j

        dx = abs(i - goal_i) # row difference
        dy = abs(j - goal_j) # column difference

        dist += dx * 2 + dy * (3 if (j > goal_j) else 1) # 3 if move right, 1 if move left

        if (i, j) != (goal_i, goal_j):
          tiles_out_of_place += 1
  
  return dist + tiles_out_of_place

def a_star_search(initial_state, goal_state):
  # Perform the A* search
  open_list = []
  closed_list = set()
  heapq.heappush(open_list, initial_state)
  
  while open_list:
    current_state = heapq.heappop(open_list)
    
    if current_state == goal_state:
      return current_state
    
    closed_list.add(current_state)
    
    for next_state in generate_next_states(current_state):
      if next_state in closed_list:
        continue
      
      next_state.g = current_state.g + 1
      next_state.h = heuristic(next_state.board, goal_state.board)
      next_state.f = next_state.g + next_state.h
      next_state.parent = current_state
      
      if next_state not in open_list:
        heapq.heappush(open_list, next_state)
        
  return None

def main():
  # Read the initial and goal states from the file
  
  
  # Solve the puzzle
  initial_state = PuzzleState(initial_state)
  goal_state = PuzzleState(goal_state)
  solution = a_star_search(initial_state, goal_state)
  
  # Output the solution
  if solution:
    path = reconstruct_path(solution)
    for board in path:
      print(board)
  else:
    print('No solution found')

if __name__ == '__main__':
  main()