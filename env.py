from collections import deque
import numpy as np
import random

class MineSweeperEnv:
    def __init__(self):
        self.game_over = False
        self.grid = np.full((14, 14), -1)
        self.mines = np.full((14, 14), 0)
        self.squares_left = 14*14
        self.generate_mines(40)
        self.observation_space = self.state().shape
        self.action_space = (14,14)  # Define the action space

    def generate_mines(self, mines):
        for i in range(mines):
            row, col = random.randint(0,13), random.randint(0,13)
            while self.mines[row][col] == 1:
                row, col = random.randint(0,13), random.randint(0,13)
            self.mines[row][col] = 1


    def reset(self):
        # Reset the game to start a new episode
        self.grid = np.full((14, 14), -1)
        self.mines = np.full((14, 14), 0)
        self.generate_mines(40)
        self.game_over = False
        self.squares_left = 14*14
        return self.state()

    def step(self, action):
        # Execute the action in the game
        row,col=action
        self.reveal(row,col)
        # Get the new state, reward, and done status
        new_state = self.state()
        reward = self.get_reward(action)
        done = self.game_over
        return new_state, reward, done, {}

    def get_reward(self, action) -> float:
        row, col = action
        if self.mines[row][col] == 1:
            self.game_over = True
            return -100
        if self.squares_left == 40:
            self.game_over = True
            return 100
        # if they just found a non mine, reward em ten.
        return 10
        # ideas:
        # Reward blocking of wins.
        # Punish for allowing wins.

    def state(self):
        return self.grid

    def close(self):
        # Close and clean up the environment
        return

    def __str__(self):
        s = "-" * (self.observation_space[1] * 2 + 1) + "\n"
        for row in self.grid:
            s += "|"
            for cell in row:
                s += str(cell) + ' | '
            s += "\n"
        s += "-" * (self.observation_space[1] * 2 + 1)
        return s

    def reveal(self, row, col):
        if self.is_valid_position(row,col):
            self.grid[row][col] = self.calculate_nearby(row,col)
            self.squares_left -= 1
    
    def calculate_nearby(self, row, col) -> int:
        count = 0
        for adjR in [-1,0,1]:
            for adjC in [-1,0,1]:
                if adjR != 0 and adjC != 0 and self.is_valid_position(adjR+row, adjC+col):
                    n_row, n_col = row+adjR, col+adjC
                    if self.mines[n_row][n_col] == 1:
                        count+=1
        return count

    def is_valid_position(self, row, col):
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])
