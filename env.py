from collections import deque
import numpy as np
import random

class Connect4Env:
    def __init__(self):
        self.game_over = False
        self.grid = np.full((14, 14), -1)
        self.mines = np.full((14, 14), 0)
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
        self.generate_mines()
        self.game_over = False

        return self.state()

    def step(self, action):
        # Execute the action in the game
        self.place(action, self.current_player)
        done = self.is_game_over()
        # Get the new state, reward, and done status
        new_state = self.state()
        reward = self.get_reward(action)
        return new_state, reward, done, {}

    def get_reward(self, action) -> float:
        if self.winner is not None:
            if self.winner == self.current_player:
                return 30
            elif self.winner != self.current_player:
                return -100
        if action
        return res
        # ideas:
        # Reward blocking of wins.
        # Punish for allowing wins.

    def state(self):
        player_ch = np.full(self.grid.shape, self.current_player)
        return np.stack((self.grid, player_ch), axis=-1)

    def close(self):
        # Close and clean up the environment
        return

    def __str__(self):
        s = "-" * (self.observation_space[1] * 2 + 1) + "\n"
        for row in self.grid:
            s += "|"
            for cell in row:
                s += str(cell) + '|'
            s += "\n"
        s += "-" * (self.observation_space[1] * 2 + 1)
        return s

    def reveal(self, row, col):
        if self.is_valid_position(row,col):
            self.grid[row][col] = self.calculate_nearby(row,col)
    
    def calculate_nearby(self, row, col) -> int:
        for adjR in [-1,0,1]:
            for adC in [-1,0,1]:
                

    def is_valid_position(self, row, col):
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])