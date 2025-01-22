# Author: Taylor Tomblin & Michael Luong
# Login: ttomblin, michalu
# Date: 1/9/2025
# Description: This program solves the 8-puzzle problem using the A* search algorithm with the Manhattan distance heuristic.
# The program reads the initial and goal states of the puzzle from a file and outputs the solution to the console.
# The program also outputs the number of nodes expanded and the number of nodes in the open list at the end of the search.

import heapq

class PuzzleState:
  def __init__(self, board, parent=None):
    self.board = board
    self.parent = parent
    self.f = 0
    self.g = 0
    self.h = 0

  def __lt__(self, other):
    return False

def generate_next_states(current_state):
  # Generate possible next states
  next_states = []
  size = len(current_state.board)

  blank_x, blank_y = next((i, j) for i, row in enumerate(current_state.board) for j, value in enumerate(row) if value == 0) # Find the index of blank tile
  directions = {
      'N': (-1, 0),
      'S': (1, 0),
      'E': (0, 1),
      'W': (0, -1)
      }
    
  for direction in directions:
    new_blank_x, new_blank_y = blank_x + directions[direction][0], blank_y + directions[direction][1]
      
    if 0 <= new_blank_x < size and 0 <= new_blank_y < size:
      new_board = [list(row) for row in current_state.board] # Converting tuple to list to swap tiles
      new_board[blank_x][blank_y], new_board[new_blank_x][new_blank_y] = new_board[new_blank_x][new_blank_y], new_board[blank_x][blank_y] # Swap blank tile position 
      next_states.append(PuzzleState(board=tuple(tuple(row) for row in new_board), parent=current_state))
  
  return next_states

def reconstruct_path(state):
  path = []

  while state:
    path.append(state)
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
  
  return dist, tiles_out_of_place

def find_zero_position(board):
    #Helper function to locate the position of 0 in a 2D tuple.
    for row_idx, row in enumerate(board.board):
        if 0 in row:
            return row_idx, row.index(0)

def a_star_search(initial_state, goal_state):
  # Perform the A* search
  open_set = []
  heapq.heappush(open_set, (0, initial_state))
  came_from = {}
  initial_state.h = heuristic(initial_state.board, goal_state)[0] + heuristic(initial_state.board, goal_state)[1]
  g_score = {initial_state: 0}
  f_score = {initial_state: g_score[initial_state] + initial_state.h}

  while open_set:
    _, current = heapq.heappop(open_set)

    if current.board == goal_state:
      return reconstruct_path(current)

    for neighbor in generate_next_states(current):
      zero_position_current = find_zero_position(current)
      zero_position_neighbor = find_zero_position(neighbor)

      # Calculate movement direction
      if zero_position_neighbor[1] == zero_position_current[1] - 1:  # Left
        move_cost = 3
      elif zero_position_neighbor[1] == zero_position_current[1] + 1:  # Right
        move_cost = 1
      else:  # Vertical (up or down)
        move_cost = 2

      tentative_g_score = g_score[current] + move_cost

      if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
        came_from[neighbor] = current
        g_score[neighbor] = tentative_g_score
        neighbor.g = g_score[neighbor]
        dist, tile_oop = heuristic(neighbor.board, goal_state)
        neighbor.h = dist + tile_oop
        neighbor.f = neighbor.g + neighbor.h
        f_score[neighbor] = neighbor.f
        print(f"Processing state with heuristic {f_score[neighbor]}: {neighbor.board}")
        heapq.heappush(open_set, (f_score[neighbor], neighbor))

  return None

def main():
  # Read the initial and goal states
  initial_state = PuzzleState(((1, 6, 2),
                               (5, 7, 8),
                               (0, 4, 3)))
  goal_state = ((7, 8, 1),
                (6, 0, 2),
                (5, 4, 3))

  path = a_star_search(initial_state, goal_state)

  # Print the path
  if path:
    print("\nPath found:")
    for state in path:
      for row in state.board:
        print(row)
      print(f'State G Score: {state.g}')
      print(f"State H Score: {state.h}")
      print()
  else:
    print("No path found")

if __name__ == "__main__":
  main()