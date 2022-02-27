from gettext import translation
from locale import currency
import numpy as np
import os
import  cv2
import copy

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

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
        # init_pos = 2,2
        self.pos = init_pos
        self.R_max = R_max
        self.T = 0

    def visualise_grid(self, file_path):
        # with open(file_path, "w")  as f:
        #     for  y in range(self.rows):
        #         for x in range(self.cols):
        #             if self.pos[0] == x and self.pos[1] == y:
        #                 f.write("|@ ")
        #                 continue
        #             if self.grid[y][x] == 1:
        #                 f.write("|# ")
        #             else:
        #                 f.write("|  ")
        #         f.write("|\n")
        grid_len = 100
        grid = 255*np.ones((grid_len*self.rows, grid_len*self.cols, 3))
        for y in range(self.rows):
            for x in range(self.cols):
                if self.pos[0] == y and self.pos[1] == x:
                    center = (x*grid_len + grid_len//2, y * grid_len+ grid_len//2)
                    cv2.circle(grid, center, grid_len//3, color=(0,255,0), thickness=10)
                    continue
                if self.grid[x][y] == 1:
                    grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 1] = 0
                    grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 2] = 0
        cv2.imwrite(file_path, grid)


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



class estimator():
    def __init__(self, X, Y, R_max, grid, initial_observation):
        self.grid = copy.deepcopy(grid)
        self.cols = X
        self.rows = Y
        self.dim =  X * Y
        self.R_max = R_max
        self.T = 0
        self.directions = {
            "N": (0, -1),
            "S": (0, 1),
            "E": (1, 0),
            "W": (-1, 0),
        }
        self.order = {
            "N": 0,
            "S": 1,
            "E": 2,
            "W": 3,
        }

        ## using initial distribution as uniform distribution
        # self.initial_distribution = np.zeros((self.dim))
        self.initial_distribution = [0.0 for i in range(self.dim)]
        count_valid_cells = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ( grid[i][j] == 0 ) :
                    count_valid_cells += 1
        for i in range(self.rows):
            for j in range(self.cols):
                if ( grid[i][j] == 0 ):
                    self.initial_distribution[i*self.rows+j] = 1/count_valid_cells

        print("Initial distribution :: ",self.initial_distribution)

        ## calculating probability P(Xt+1|Xt) = Sigma P(Xt+1|Ut,Xt) P(Ut|Xt)
        # self.transition_probability = np.zeros((self.dim,self.dim))
        self.transition_probability = [[0.0 for i in range(self.dim)] for j in range(self.dim)]
        for i in range  (self.dim):
            x = i % self.rows
            y = i // self.rows
            if grid[y][x] == 0:
                moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                allowed_moves = []
                for j in range(len(moves)):
                    m = moves[j]
                    if self.grid[y+m[1]][x+m[0]] == 0:
                        allowed_moves.append(m)
                moves_possible = len(allowed_moves)
                for m in allowed_moves:
                    y_new = y + m[1] 
                    x_new = x + m[0]
                    self.transition_probability[y_new * self.rows + x_new][i] = 1/moves_possible 
        
        print("Transition_probability :: ",self.transition_probability) 

        ## calculating probability P(Ot|Xt)  1 == Close , 0 == Far encoded as 1001 
        # self.observation_probablity = np.zeros((16,self.dim))
        self.observation_probablity = [[0.0 for i in range(self.dim)] for j in range(16)]

        for i in range (self.dim):
            x = i % self.rows
            y = i // self.rows
            if self.grid[y][x] == 0:
                for j in range(16):
                    self.observation_probablity[j][i] = 1.0
                for sensor in ["N", "S", "E", "W"]:
                    ij = 1
                    dir = self.directions[sensor]
                    while self.grid[y+ij*dir[1]][x + ij*dir[0]] == 0:
                        ij += 1
                    prob = 1.0
                    if ij < self.R_max:
                        prob = (ij-1)/(self.R_max - 1)
                    for j in range(16):
                        if( ( j & (1 << self.order[sensor] ) ) > 0 ) :
                            if i == 6:
                                print( j , sensor)
                            self.observation_probablity[j][i] *= (1.0 - prob)
                        else:
                            if ( j == 0 ):
                                print(i , prob, ij)
                            self.observation_probablity[j][i] *= prob
        
        print("Observation_probability :: ",self.observation_probablity) 
        
        self.init_observation = self.encode_observation(initial_observation)
        print(self.init_observation)
        self.observations = [self.init_observation]
        current_estimate = self.initial_distribution
        sum = 0.0
        for i in range(self.dim):
            x = i % self.rows
            y = i // self.rows
            current_estimate[i] *= self.observation_probablity[self.init_observation][i]
            sum += current_estimate[i]
        for i in range(self.dim):
            x = i % self.rows
            y = i // self.rows
            current_estimate[i] /= sum

        print("Current_Estimate :: ",current_estimate)
        
        self.estimates = [current_estimate]
        self.viterbi_values = [current_estimate]
        self.viterbi_best_prev_states = [[None for i in range(self.dim)]]
        self.viterbi_sequence = [argmax(self.viterbi_values[-1])]
    
    def encode_observation(self,observe):
        observation_encoded = 0
        for sensor in ["N", "S", "E", "W"]:
            if observe[sensor] == "Close" :
                observation_encoded += (1<<(self.order[sensor]))
        return observation_encoded
            

    def update(self,current_observation):
        encoded_observation = self.encode_observation(current_observation)
        self.observations.append(encoded_observation)
        self.T += 1
        self.update_current_estimate()
        self.update_most_likely_estimate()

    def update_most_likely_estimate(self):
        new_viterbi_values = [0.0 for i in range(self.dim)]
        new_viterbi_best_prev_state = [None for i in range(self.dim)]
        for cur_state in range(self.dim):
            max_tr_prob , prev_st_selected = None , None
            for prev_state in range(self.dim):
                tr_prob = self.viterbi_values[-1][prev_state] * self.transition_probability[cur_state][prev_state]
                if max_tr_prob == None or tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_state
            new_viterbi_values[cur_state] = max_tr_prob * self.observation_probablity[self.observations[-1]][cur_state]
            new_viterbi_best_prev_state[cur_state] = prev_st_selected
        self.viterbi_values.append(new_viterbi_values)
        self.viterbi_best_prev_states.append(new_viterbi_best_prev_state)
        self.update_viterbi_sequence()
    
    def update_viterbi_sequence(self):
        self.viterbi_sequence = []
        last_ver = argmax(self.viterbi_values[-1])
        cur_time = self.T
        while( cur_time != -1):
            self.viterbi_sequence.append(last_ver)
            last_ver = self.viterbi_best_prev_states[cur_time][last_ver]
            cur_time -= 1
        self.viterbi_sequence = self.viterbi_sequence[::-1]

    def update_current_estimate(self):
        current_values = [0.0 for i in range(self.dim)]
        current_value_sum = 0.0
        for cur_state in range(self.dim):
            tr_prob =  0.0
            for prev_state in range(self.dim):
                tr_prob += self.estimates[-1][prev_state] * self.transition_probability[cur_state][prev_state]
            current_values[cur_state] = tr_prob * self.observation_probablity[self.observations[-1]][cur_state]
            current_value_sum += current_values[cur_state]

        # print(current_value_sum)

        assert(current_value_sum != 0.0)

        for cur_state in range(self.dim):
            current_values[cur_state] /= current_value_sum
            
        self.estimates.append(current_values)

    def get_current_estimate(self):
        return (argmax(self.estimates[-1])//self.rows , argmax(self.estimates[-1]) % self.rows )
        

os.makedirs("grids_img", exist_ok=True)
os.makedirs("log_likelihoods", exist_ok=True)
os.makedirs("viterbi", exist_ok=True)


X , Y = 20 , 20
R_max = 5
n_walls = 50
model = RobotModel(X, Y, R_max,n_walls)

observation = model.make_observation()
# observation = {'N': 'Far', 'S': 'Far', 'E': 'Close', 'W': 'Far'}
print( observation )
estimate = estimator(X, Y, R_max,copy.deepcopy(model.grid),observation)
model.visualise_grid("grids_img/{}.png".format(-1))
for i in range(100):
    model.visualise_grid("grids_img/{}.png".format(i))
    model.make_random_move()
    print(model.pos)
    observation = model.make_observation()
    print( observation )
    estimate.update(observation)
    print(estimate.estimates[-1])
    grid_len = 100
    grid = np.zeros((grid_len*model.rows, grid_len*model.cols, 3))
    for y in range(model.rows):
        for x in range(model.cols):
            if model.grid[x][y] == 1:
                grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 0] = 255
            else:
                grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 2] = 255*estimate.estimates[-1][x*model.rows + y]
            if model.pos[0] == y and model.pos[1] == x:
                center = (x*grid_len + grid_len//2, y * grid_len+ grid_len//2)
                cv2.circle(grid, center, grid_len//3, color=(0,255,0), thickness=10)

    cv2.imwrite("log_likelihoods/{}.png".format(i), grid)


    grid_len = 100
    grid = np.zeros((grid_len*model.rows, grid_len*model.cols, 3))
    for y in range(model.rows):
        for x in range(model.cols):
            if model.grid[x][y] == 1:
                grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 0] = 255
            
    k =  10
    k = min(k, len(estimate.viterbi_sequence))
    for v_iter in range(k):
        pos = estimate.viterbi_sequence[len(estimate.viterbi_sequence) - k + v_iter]
        y, x = pos%model.rows,  pos//model.rows
        grid[y*grid_len:(y+1)*grid_len, x*grid_len:(x+1)*grid_len, 2] = 255*(v_iter/k)
    for y in range(model.rows):
        for x in range(model.cols):
            if model.pos[0] == y and model.pos[1] == x:
                center = (x*grid_len + grid_len//2, y * grid_len+ grid_len//2)
                cv2.circle(grid, center, grid_len//3, color=(0,255,0), thickness=10)


    cv2.imwrite("viterbi/{}.png".format(i), grid)


    # print(estimate.get_current_estimate())