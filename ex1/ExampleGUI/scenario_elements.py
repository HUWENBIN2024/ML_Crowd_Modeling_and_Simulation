import math

import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import networkx as nx
import copy

class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed

        self.status = 'walking' # walking or finished
        self._before_schedule = 0

        self._track = []
        self.step_length = desired_speed * 1
        self.move_dist = 0
        self.accumulate_steps = 1


    @property
    def position(self):
        return self._position
    
    @property
    def track(self):     #visualize track of pedestrains
        return self._track

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width 
            and 0 <= y + self._position[1] < scenario.height 
            and (x , y) != (0, 0)

        ]
    

    def get_next_position(self, scenario):

        neighbors = self.get_neighbors(scenario)
        cost = copy.deepcopy(scenario.cost)
        scenario.individual_repulse_force(cost, self._position[0], self._position[1], sign=-1)
        next_cell_distance = cost[self._position[0]][self._position[1]]
        next_pos = self._position
        x, y = self._position
        self._track.append(self._position)
        for (n_x, n_y) in neighbors:
            if next_cell_distance > cost[n_x, n_y] and abs(next_cell_distance - cost[n_x, n_y]) > 1e-10:
                next_pos = (n_x, n_y)
                next_cell_distance = cost[n_x, n_y]
            elif abs(next_cell_distance - cost[n_x, n_y]) < 1e-10 and (n_x - x)**2 + (n_y - y)**2 == 1:
                next_pos = (n_x, n_y)
                next_cell_distance = cost[n_x, n_y]             

        return next_pos


    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        
        :param scenario: The current scenario instance.
        """
        next_pos = self.get_next_position(scenario)
        self.move_dist = ((self.position[0] - next_pos[0]) ** 2 + (self.position[1] - next_pos[1]) ** 2) ** (1/2)
        if self.move_dist <= self.step_length * self.accumulate_steps:
            while self.move_dist <= self.step_length * self.accumulate_steps:
                self._position = next_pos
                for tar in scenario.target_list:
                    if (self._position[0], self._position[1]) == (tar[0], tar[1]):
                        self.status = 'finished'     
                        return
                next_pos = self.get_next_position(scenario)
                if (self.position[0] - next_pos[0]) ** 2  == (self.position[1] - next_pos[1]):
                    break
                self.move_dist += ((self.position[0] - next_pos[0]) ** 2 + (self.position[1] - next_pos[1]) ** 2 ) ** (1/2)
            self.move_dist = 0 
            self.accumulate_steps = 1
        else:
            self.accumulate_steps += 1


                   
        

class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0), # red
        'TARGET': (0, 0, 255),     # violet
        'OBSTACLE': (0, 255, 0) , # yellow
        'TRACK':(128,125,125) ,    #grey
      
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }

    def __init__(self, width, height, args):
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")
        self.r_max = args.r_max
        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians = []
        self.target_list = []
        self.distance_mode = args.distance_mode
        self.cost = np.zeros((width, height))

        self.target_distance_grids = self.recompute_target_distances()
      
    def get_neighbors(self, i, j):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + i), int(y + j))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + i < self.width 
            and 0 <= y + j < self.height 
            and np.abs(x) + np.abs(y) > 0
        ]

    def recompute_target_distances(self):
        '''
        Choose which distance mode.
        '''
        if self.distance_mode == 'dijkstra':
            self.target_distance_grids = self.walking_distance()
        else:
            self.target_distance_grids = self.euclidean_distance()
        return self.target_distance_grids
    
    def get_targets(self, ):
        '''
        Get targets from self.grid .
        '''
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        return targets

    def walking_distance(self, ):
        """
        Computes the shortest manhattan distance from every grid point to targets using Dijkstra algorithm.
        """
        targets = self.get_targets()
        if len(targets) == 0:
            return np.zeros((self.width, self.height))
        
        G = self.grid_to_graph(self.grid)
        target_id_list = [self.cod2id(tar[0], tar[1]) for tar in self.target_list]
        dijkstra_dist = nx.multi_source_dijkstra_path_length(G, target_id_list)
        dist = np.full((self.width, self.height), np.inf)
        for i in range(self.width):
            for j in range(self.height):
                try:
                    dist[i, j] = dijkstra_dist[self.cod2id(i, j)]
                except:
                    pass
        return dist

    def grid_to_graph(self, grid):
        '''
        Build a graph by our grid data structure, using networkx package.

        Args: 
            grid: a 2d numpy array that save status in each grid. (normally, it's the same of self.grid)
        Return:
            An undirectional graph whose vertises represent grids without obstacles.
        '''
        obstacle_list = []
        G = nx.Graph()
        G = G.to_undirected()

        # add nodes
        for id in range(self.width * self.height):
            G.add_node(id)
        nb_list = self.get_neighbors(25, 25) # a list of neighbour coordinates

        # add edges
        for i in range(self.width):
            for j in range(self.height):
                id = self.cod2id(i, j)
                if grid[i, j] == self.NAME2ID['OBSTACLE']:
                    obstacle_list.append(id)
                nb_list = self.get_neighbors(i, j) # a list of neighbour coordinates
                for k, nb in enumerate(nb_list):
                    nb_id = self.cod2id(nb[0], nb[1])
                    if k in [0, 2, 6, 8]:
                        G.add_edge(id, nb_id, weight=2**(1/2))
                    else:
                        G.add_edge(id, nb_id, weight=1)
        G.remove_nodes_from(obstacle_list)
        return G
    
    def id2cod(self, id):
        '''
        Convert id to coordinate. 

        Args:
            id: the id we assign to a specific pedestrian. Range: 0 ~ w*h -1. Eg. id = 100.
        '''
        return id / self.width, id % self.width

    def cod2id(self, i, j):
        '''
        Convert coordinate to id.

        Args:
            coordinate: the coordinate of specific pedestrian. 
        '''    
        return j * self.width + i
        
    def euclidean_distance(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = self.get_targets()
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        return distances.reshape((self.width, self.height))

    def is_inbound(self, i, j):
        '''
        Whether (i, j) is our scenario boundary.
        '''
        return -1 < i and i < self.width and -1 < j and j < self.height

    def calculate_cost(self, ):
        '''
        Calculate the forse among individuals
        '''
        ped_cost = np.zeros((self.width, self.height))
        for ped in self.pedestrians:
            x, y = ped._position
            self.individual_repulse_force(ped_cost, x, y)
        self.cost = ped_cost + self.target_distance_grids

    def individual_repulse_force(self, cost_matrix, x, y, sign=1):
        '''
        Influnce brought by one individual.
        The implementation idea comes from Newton's Third Law.

        Args: 
            cost_matrix: a numpy matrix that save cost values for each grid.
            x: x coordinate.
            y: y coordinate.
            sign: positive or negative.
        '''
        r_max = self.r_max
        for i in range(-r_max, r_max+1):
            for j in range(-r_max, r_max+1):
                if i**2 + j**2 < r_max**2 and self.is_inbound(x+i, y+j):
                    cost_matrix[x+i, y+j] += sign * np.exp(1 / (i**2 + j**2 - r_max**2))

    def update_step(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        self.calculate_cost()
        for pedestrian in self.pedestrians:
            if pedestrian.status == 'finished':
                self.pedestrians.remove(pedestrian)
            else:
                pedestrian.update_step(self)

    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self,canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                if target_distance > 1e10:
                    target_distance = 1e5
                pix[x, y] = (max(93, min(255, int(10 * target_distance) - 0 * 139)),
                             max(71, min(255, int(10 * target_distance) - 1 * 139)),
                             max(130, min(255, int(10 * target_distance) - 2 * 139)))
        
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(self.grid[x, y])
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            for [x,y] in pedestrian.track:
                if not self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    pix[x, y] = Scenario.NAME2COLOR['TRACK']
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)
