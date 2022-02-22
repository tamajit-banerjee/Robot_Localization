import numpy as np
import os

class RobotModel():
    def __init__(self, X, Y, R_max=5, n_walls=40):
        self.grid = [[0 for i in range(X)] for j in range(Y)]
        self.cols = X
        self.rows = Y
        for i in range(X):
            self.add_wall(i, 0)
            self.add_wall(i, -1)
        for i in range(Y):
            self.add_wall(0, i)
            self.add_wall(-1, i)
        for i in range(n_walls):
            x, y = np.random.randint(1, X-1), np.random.randint(1, Y-1)
            self.add_wall(x, y)



        
        init_pos = np.random.randint(1, X-1), np.random.randint(1, Y-1)
        while self.grid[init_pos[1]][init_pos[0]] == 1:
            init_pos = np.random.randint(1, X-1), np.random.randint(1, Y-1)
        self.pos = init_pos
        self.R_max = R_max

    def visualise_grid(self, file_path):
        with open(file_path, "w")  as f:
            for  y in range(self.rows):
                for x in range(self.cols):
                    if self.pos[0] == x and self.pos[1] == y:
                        f.write("|@ ")
                        continue
                    if self.grid[y][x] == 1:
                        f.write("|# ")
                    else:
                        f.write("|  ")
                f.write("|\n")


    def add_wall(self, x, y):
        self.grid[y][x] = 1

    def make_random_move(self):
        moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        allowed_moves = []
        for i in range(len(moves)):
            x, y = self.pos
            m = moves[i]
            if self.grid[y+m[1]][x+m[0]] == 0:
                allowed_moves.append(i)
        move = moves[np.random.choice(allowed_moves)]
        self.pos = (self.pos[0] + move[0], self.pos[1] + move[1])

    def make_observation(self):
        x, y = self.pos
        directions = {
            "N": (0, -1),
            "S": (0, 1),
            "E": (1, 0),
            "W": (-1, 0),
        }
        obs = {
            "N": None,
            "S": None,
            "E": None,
            "W": None,
        }
        for sensor in ["N", "S", "E", "W"]:
            i = 0
            dir = directions[sensor]
            while self.grid[y+i*dir[1]][x + i*dir[0]] == 0:
                i += 1
            
            if i>=self.R_max or self.R_max == 1:
                obs[sensor] = "Far"
            else:
                prob = 1 - (i-1)/(self.R_max - 1)
                obs[sensor] = np.random.choice(["Close", "Far"], p=[prob, 1-prob])
        return obs





        

os.makedirs("grids", exist_ok=True)


model = RobotModel(7, 7, n_walls=4, R_max=2)
for i in range(100):
    model.visualise_grid("grids/{}.txt".format(i))
    model.make_random_move()
    print(model.pos)
    print(model.make_observation())