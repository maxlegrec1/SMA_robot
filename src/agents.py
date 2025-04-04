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
            #print(self.knowledge['internal_map'][:,:,0])
            return Move(next_cell_direction)
        
        if self.state == "FINDING_WASTE":
            for next_cell in possible_next_cell:
                if self.get_radioactivity((self.knowledge['agent_x'],self.knowledge['agent_y'])) > color_dict[self.color]/3 +1e-10:
                    return Move("WEST")
            #if on the boundary, random walk north and south
            #print(possible_next_cell)
            possible_next_cell_dir_shuffled = [inv_direction_dict[cell] for cell in possible_next_cell]
            random.shuffle(possible_next_cell_dir_shuffled)
            for dir in possible_next_cell_dir_shuffled:
                if dir == "NORTH" or dir == "SOUTH":
                    return Move(dir)


        if self.state == "TRANSPORTING":
            for next_cell in possible_next_cell:
                if inv_direction_dict[next_cell] == "EAST":
                    return Move("EAST")
            #go up for red robot 
            if self.color == "red":   
                for next_cell in possible_next_cell:
                    if inv_direction_dict[next_cell] == "NORTH":
                        return Move("NORTH")
            self.state = "FINDING_WASTE"
            #print(self.knowledge['transporting'])
            return Drop(self.knowledge['transporting'][0])

    def update_knowledge(self,observation):
        for (x,y) in observation:
            if not self.in_map((x + self.knowledge['agent_x'],y + self.knowledge['agent_y'])) and abs(x) + abs(y) == 1:  #second condition makes sure that its N,S,W,E
                self.knowledge['internal_map'] = expand_grid(self.knowledge['internal_map'],(x,y))
                self.knowledge['agent_x'] += max(0,-x)
                self.knowledge['agent_y'] += max(0,-y)
        #finish updating knowledge
        #add one age to all squares 
        self.knowledge['internal_map'][:,:,-1] += 1 
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
                    self.knowledge['internal_map'][x + self.knowledge['agent_x'],y + self.knowledge['agent_y'],2 + color_dict[agent.color]] += 1 
                    self.knowledge['waste_color'][agent.unique_id] = agent.color
        self.knowledge['last_observation'] = observation
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
