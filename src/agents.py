"""
  ____  __  __    _              _____    _        __  ____     __
 / ___||  \/  |  / \            | ____|  / \      |  \/  \ \   / /
 \___ \| |\/| | / _ \    _____  |  _|   / _ \     | |\/| |\ \ / / 
  ___) | |  | |/ ___ \  |_____| | |___ / ___ \ _  | |  | | \ V /  
 |____/|_|  |_/_/   \_\         |_____/_/   \_( ) |_|  |_|  \_/   
                                              |/                  
Authors:
   Maxime Vanderbeken
   Etienne Andrier

Date : 2025-03-19

License:
   This file is open source and may be freely used and modified,
   provided that proper credit is given to the original authors.
"""
from mesa import Agent
from .action import Move, Drop, NoneAction
import numpy as np
from scipy.ndimage import binary_dilation
from .variables import color_dict,direction_dict,inv_direction_dict
from .knowledge_expansion import expand_grid
import random

class BaseAgent(Agent):
    def __init__(self,model):
        super().__init__(model)
        self.model = model
        self.knowledge = {}

    def deliberate(self):
        '''Return an action based on the knowledge'''
        return NoneAction()

    def step_agent(self,observation):
        self.update_knowledge(observation) 
        action = self.deliberate() 
        #model perfoms action , technically the agent just 'asks'
        self.model.do(self,action)
    def update_knowledge(self,observation):
        pass
class RadioactivityAgent(BaseAgent):
    def __init__(self,model,radioactivity):
        super().__init__(model)
        self.radioactivity = radioactivity
        
class RobotAgent(BaseAgent):
    def __init__(self,model,color):
        super().__init__(model)
        self.color = color
        if self.color == "red":
            self.max_allowed_radioactivity = 2
        else:
            self.max_allowed_radioactivity = (color_dict[self.color] + 1) / 3 - 1e-10 
        self.knowledge['transporting'] = []
        self.knowledge['internal_map'] = np.zeros((1,1,6))
        #on each square of the grid is a tuple (radioactivity, num_agents on that square counting (self), and number of each waste type on that cell (green,yellow,red), age of information
        self.knowledge['internal_map'][0,0,0] = color_dict[color]
        self.knowledge['internal_map'][0,0,1] = 1 #self
        self.knowledge['agent_x'] = 0
        self.knowledge['agent_y'] = 0
        self.knowledge['waste_color'] = {}
        self.state = "FINDING_WASTE"
    def deliberate(self):
        #print(self.knowledge['internal_map'].shape)
        ''' What should the agent do (what action) depending on self.knowledge'''
        #COMPUTE STATE FROM KNOWLEDGE
        #if transporting a waste of higher color than self, transporting to east
        if len(self.knowledge['transporting']) >= 1 and color_dict[self.knowledge['waste_color'][self.knowledge['transporting'][0]]] > color_dict[self.color]:
            self.state = "TRANSPORTING"
        #if transporting red and is color is red
        if self.color == "red" and len(self.knowledge['transporting']) >= 1 and color_dict[self.knowledge['waste_color'][self.knowledge['transporting'][0]]] >= color_dict[self.color]:
             self.state = "TRANSPORTING"
        possible_next_cell = []
        for (x,y) in self.knowledge['last_observation']:
            if (x,y) != (0,0) and self.get_radioactivity((x + self.knowledge['agent_x'], y + self.knowledge['agent_y'])) <= self.max_allowed_radioactivity:
                possible_next_cell.append((x,y))
        #WHAT TO DO FOR EACH STATE
        #print(self.state,possible_next_cell,self.color,self.max_allowed_radioactivity)
        if self.state == "FINDING_WASTE" and self.color == "green":
            #print(possible_next_cell)
            next_cell = random.choice(possible_next_cell)
            next_cell_direction = inv_direction_dict[next_cell]
            x,y = next_cell
            self.knowledge['agent_x'] += x
            self.knowledge['agent_y'] += y
            return Move(next_cell_direction)
        
        if self.state == "FINDING_WASTE":
            for next_cell in possible_next_cell:
                if self.get_radioactivity((self.knowledge['agent_x'],self.knowledge['agent_y'])) > color_dict[self.color]/3 +1e-10:
                    x,y = direction_dict["WEST"]
                    self.knowledge['agent_x'] += x
                    self.knowledge['agent_y'] += y
                    return Move("WEST")
            #if on the boundary, random walk north and south
            #print(possible_next_cell)
            possible_next_cell_dir_shuffled = [inv_direction_dict[cell] for cell in possible_next_cell]
            random.shuffle(possible_next_cell_dir_shuffled)
            for dir in possible_next_cell_dir_shuffled:
                if dir == "NORTH" or dir == "SOUTH":
                    x,y = direction_dict[dir]
                    self.knowledge['agent_x'] += x
                    self.knowledge['agent_y'] += y
                    return Move(dir)


        if self.state == "TRANSPORTING":
            for next_cell in possible_next_cell:
                if inv_direction_dict[next_cell] == "EAST":
                    x,y = direction_dict["EAST"]
                    self.knowledge['agent_x'] += x
                    self.knowledge['agent_y'] += y
                    return Move("EAST")
            #go up for red robot 
            if self.color == "red":   
                for next_cell in possible_next_cell:
                    if inv_direction_dict[next_cell] == "NORTH":
                        x,y = direction_dict["NORTH"]
                        self.knowledge['agent_x'] += x
                        self.knowledge['agent_y'] += y
                        return Move("NORTH")
            self.state = "FINDING_WASTE"
            #print(self.knowledge['transporting'])
            return Drop(self.knowledge['transporting'][0])

    def update_knowledge(self,observation):
        for (x,y) in observation:
            if not self.in_map((x + self.knowledge['agent_x'],y + self.knowledge['agent_y'])) and abs(x) + abs(y) == 1:  #second condition makes sure that its N,S,W,E
                print(f"{self.unique_id},{self.knowledge['internal_map'].shape}, {self.knowledge['agent_x']+x}, {self.knowledge['agent_y']+y}")
                self.knowledge['internal_map'] = expand_grid(self.knowledge['internal_map'],(x,y))
                self.knowledge['agent_x'] += max(0,-x)
                self.knowledge['agent_y'] += max(0,-y)
        #finish updating knowledge
        #add one age to all squares except fog of war
        mask = self.knowledge['internal_map'][:, :, -1] != -1
        self.knowledge['internal_map'][:, :, -1][mask] += 1 
        #print(self.knowledge["internal_map"][:,:,-1])
        for (x,y) in observation:
            #first, reset information of that square
            self.knowledge['internal_map'][x + self.knowledge['agent_x'],y + self.knowledge['agent_y'],:] = 0
            #fill in knowledge
            for agent in observation[(x,y)]:
                if isinstance(agent,RadioactivityAgent):
                    self.knowledge['internal_map'][x + self.knowledge['agent_x'],y + self.knowledge['agent_y'],0] = agent.radioactivity

                if isinstance(agent,RobotAgent):
                    self.knowledge['internal_map'][x + self.knowledge['agent_x'],y + self.knowledge['agent_y'],1] +=1

                if isinstance(agent,WasteAgent):
                    if not (agent.picked_up or agent.arrived): #don't notice wastes that are picked up
                        self.knowledge['internal_map'][x + self.knowledge['agent_x'],y + self.knowledge['agent_y'],2 + color_dict[agent.color]] += 1 
                    self.knowledge['waste_color'][agent.unique_id] = agent.color
        self.knowledge['last_observation'] = observation
        assert self.knowledge['internal_map'][self.knowledge['agent_x'],self.knowledge['agent_y'],1] >= 1
    def in_map(self,square):
        x,y = square
        len_x,len_y = self.knowledge['internal_map'].shape[:2]
        if 0 <= x and x < len_x and 0 <= y and y < len_y:
            return True
        return False
    
    def pickup(self,agent_id):
        self.knowledge['transporting'].append(agent_id)

    def drop(self,agent_id):
        idx = self.knowledge['transporting'].index(agent_id)
        self.knowledge['transporting'].pop(idx)

    def get_radioactivity(self,square):
        #no fail safe yet
        return self.knowledge['internal_map'][square[0],square[1],0]

        
class WasteAgent(BaseAgent):
    def __init__(self,model,color):
        super().__init__(model)
        self.color = color
        self.picked_up = False
        self.arrived = False


class RefinedAgent(RobotAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    def deliberate(self):
        ''' What should the agent do (what action) depending on self.knowledge'''
        #COMPUTE STATE FROM KNOWLEDGE
        #if transporting a waste of higher color than self, transporting to east
        if len(self.knowledge['transporting']) >= 1 and color_dict[self.knowledge['waste_color'][self.knowledge['transporting'][0]]] > color_dict[self.color]:
            self.state = "TRANSPORTING"
        #if transporting red and is color is red
        if self.color == "red" and len(self.knowledge['transporting']) >= 1 and color_dict[self.knowledge['waste_color'][self.knowledge['transporting'][0]]] >= color_dict[self.color]:
             self.state = "TRANSPORTING"
        possible_next_cell = []
        for (x,y) in self.knowledge['last_observation']:
            if (x,y) != (0,0) and self.get_radioactivity((x + self.knowledge['agent_x'], y + self.knowledge['agent_y'])) <= self.max_allowed_radioactivity:
                possible_next_cell.append((x,y))
        #WHAT TO DO FOR EACH STATE
        #print(self.state,possible_next_cell,self.color,self.max_allowed_radioactivity)
        if self.state == "FINDING_WASTE":
            #if a waste of color is available in knowledge map, go get it.
            #Otherwise, explore.

            waste_map_agent_color = self.knowledge["internal_map"][:,:,2 + color_dict[self.color]]
            candidate_squares = (waste_map_agent_color > 0)
            #print(candidate_squares)
            if candidate_squares.any():
                #there is a waste in sight. Go for it !
                coordinates = np.argwhere(candidate_squares)
                # Convert to list of tuples (x, y)
                coordinates_list = [tuple(coord) for coord in coordinates]
                x_agent,y_agent = self.knowledge['agent_x'],self.knowledge['agent_y']
                coordinates_list = [(x-x_agent,y-y_agent) for (x,y) in coordinates_list]
                distances = [x*x + y*y for (x,y) in coordinates_list]
                closest = min(enumerate(distances), key=lambda x: x[1])[0]
                target_cell = coordinates_list[closest]
                x,y = self.normalize_direction(target_cell)
                target_dir = inv_direction_dict[(x,y)]
                self.knowledge['agent_x'] += int(x)
                self.knowledge['agent_y'] += int(y)
                return Move(target_dir)
            else:
                #nothing in sight, explore !
                age_map = self.knowledge['internal_map'][:,:,-1]
                if self.color == "red":
                    radioactivities = self.knowledge['internal_map'][:,:,0] >= (color_dict[self.color]/3 + 1e-10)
                else:
                    radioactivities = (self.knowledge['internal_map'][:,:,0] >= (color_dict[self.color]/3 + 1e-10)) & (self.knowledge['internal_map'][:,:,0] <= ((color_dict[self.color] +1)/3 - 1e-10))
                
                radioactivities = binary_dilation(radioactivities,structure=np.ones((3, 3)))

                extended_age_map = np.where(radioactivities,age_map,0)
                if -1 in extended_age_map:
                    target =-1
                    coordinates = np.argwhere(extended_age_map == target)
                    # Convert to list of tuples (x, y)
                    coordinates_list = [tuple(coord) for coord in coordinates if tuple(coord)!= (self.knowledge['agent_x'],self.knowledge['agent_y'])]
                    x_agent,y_agent = self.knowledge['agent_x'],self.knowledge['agent_y']
                    coordinates_list = [(x-x_agent,y-y_agent) for (x,y) in coordinates_list]
                    distances = [x*x + y*y for (x,y) in coordinates_list]
                    closest = min(enumerate(distances), key=lambda x: x[1])[0]
                    target_cell = coordinates_list[closest]
                    x,y = self.normalize_direction(target_cell)
                    target_dir = inv_direction_dict[(x,y)]
                    self.knowledge['agent_x'] += int(x)
                    self.knowledge['agent_y'] += int(y)
                    return Move(target_dir)
                else:
                    i_indices, j_indices = np.meshgrid(np.arange(age_map.shape[0]), np.arange(age_map.shape[1]), indexing='ij')
                    distance_map = np.abs(i_indices - self.knowledge['agent_x']) + np.abs(j_indices - self.knowledge['agent_y'])
                    target = np.max(np.where(radioactivities,age_map-distance_map,-10000))
                    if target <= 0:
                        if self.get_radioactivity((self.knowledge['agent_x'],self.knowledge['agent_y'])) < color_dict[self.color]/3 +1e-10: #Si deja sur la frontière, interdiction d'aller plus à gauche
                            #remove westish from possibilities
                            possible_next_cell = [(x,y) for (x,y) in possible_next_cell if x>= 0]
                        x,y = random.sample(possible_next_cell,1)[0]
                        target_dir = inv_direction_dict[(x,y)]
                        self.knowledge['agent_x'] += int(x)
                        self.knowledge['agent_y'] += int(y)
                        return Move(target_dir) 
                    else:
                        coordinates = np.argwhere(age_map-distance_map == target)
                        # Convert to list of tuples (x, y)
                        coordinates_list = [tuple(coord) for coord in coordinates if tuple(coord)!= (self.knowledge['agent_x'],self.knowledge['agent_y'])]
                        x_agent,y_agent = self.knowledge['agent_x'],self.knowledge['agent_y']
                        coordinates_list = [(x-x_agent,y-y_agent) for (x,y) in coordinates_list]
                        distances = [x*x + y*y for (x,y) in coordinates_list]
                        closest = min(enumerate(distances), key=lambda x: x[1])[0]
                        target_cell = coordinates_list[closest]
                        x,y = self.normalize_direction(target_cell)
                        target_dir = inv_direction_dict[(x,y)]
                        self.knowledge['agent_x'] += int(x)
                        self.knowledge['agent_y'] += int(y)
                        return Move(target_dir)

                

        if self.state == "TRANSPORTING":
            for next_cell in possible_next_cell:
                if inv_direction_dict[next_cell] == "EAST":
                    x,y = direction_dict["EAST"]
                    self.knowledge['agent_x'] += x
                    self.knowledge['agent_y'] += y
                    return Move("EAST")
            #go up for red robot 
            if self.color == "red":   
                for next_cell in possible_next_cell:
                    if inv_direction_dict[next_cell] == "NORTH":
                        x,y = direction_dict["NORTH"]
                        self.knowledge['agent_x'] += x
                        self.knowledge['agent_y'] += y
                        return Move("NORTH")
            self.state = "FINDING_WASTE"
            #print(self.knowledge['transporting'])
            return Drop(self.knowledge['transporting'][0])


    def normalize_direction(self,target_cell):
        x, y = target_cell
        return (
            0 if x == 0 else x / abs(x),
            0 if y == 0 else y / abs(y)
        )