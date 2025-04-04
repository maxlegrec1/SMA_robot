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
from mesa.datacollection import DataCollector
import mesa
from mesa import Model
from .agents import RadioactivityAgent, RobotAgent, WasteAgent
import numpy as np
from .action import Move, Drop, NoneAction
from .variables import color_dict,direction_dict,inv_direction_dict
# Add the following helper functions to model.py to count different waste types
def count_green_waste(model):
    return sum(1 for agent in model.agents if isinstance(agent, WasteAgent) and agent.color == 'green')

def count_yellow_waste(model):
    return sum(1 for agent in model.agents if isinstance(agent, WasteAgent) and agent.color == 'yellow')

def count_red_waste(model):
    return sum(1 for agent in model.agents if isinstance(agent, WasteAgent) and agent.color == 'red')

def count_disposed_waste(model):
    # Count waste that has been properly disposed (you may need to add a flag to track this)
    return model.disposed_waste_count

def next_color(color):
    if color=='green':
        return 'yellow'
    if color=="yellow":
        return "red"

class WasteRetrievalModel(Model):
    def __init__(self,
                 num_green = 1,
                 num_yellow =1,
                 num_red = 1,
                 num_waste = 1,
                 width = 30,
                 height = 30,
                 seed=None):
        super().__init__(seed=seed)
        if(width%3!=0):
            raise Exception("The indicated width is not a multiple of 3")
        self.width = width
        self.height = height
        self.num_green = num_green
        self.num_yellow = num_yellow
        self.num_red = num_red
        self.num_waste = num_waste
        self.grid = mesa.space.MultiGrid(width, height,torus = False)
        #self.schedule = mesa.time.RandomActivation(self)
        self.disposed_waste_count = 0
        # Add the datacollector
        self.datacollector = DataCollector(
            model_reporters={
                "Green Waste": count_green_waste,
                "Yellow Waste": count_yellow_waste,
                "Red Waste": count_red_waste,
                "Disposed Waste": count_disposed_waste
            }
        )
        self.running = False # Simulation starts paused

        self.robot_agents = []
        self.waste_agents = []
        self.radioactivity_agents = []
        self.initialize_agents()
        self.datacollector.collect(self)

    def initialize_agents(self):
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.robot_agents = []
        self.waste_agents = []
        self.radioactivity_agents = []
        self.disposed_waste_count = 0
        # initialize the radioactivity agents
        for zone in range(3):
            for i in range(zone*self.width//3,(zone+1)*self.width//3):
                for j in range(self.height):
                    if i==self.width - 1 and j == self.height - 1:
                        # Red waste cell
                        agent = RadioactivityAgent(self,radioactivity=2)
                        self.grid.place_agent(agent, (i, j))
                        self.radioactivity_agents.append(agent)
                    else:
                        agent = RadioactivityAgent(self,radioactivity=0.33*zone + np.random.random()/3)
                        self.grid.place_agent(agent, (i, j))
                        self.radioactivity_agents.append(agent)


        random_pos_zone1 = set()
        while len(random_pos_zone1) < self.num_waste + self.num_green:
            random_pos_zone1.add((np.random.randint(self.width//3),np.random.randint(self.height)))
        random_pos_zone1 = list(random_pos_zone1)

        random_pos_zone2 = set()
        while len(random_pos_zone2) < self.num_yellow:
            random_pos_zone2.add((np.random.randint(self.width//3,2*self.width//3),np.random.randint(self.height)))
        random_pos_zone2 = list(random_pos_zone2)

        random_pos_zone3 = set()
        while len(random_pos_zone3) < self.num_red:
            random_pos_zone3.add((np.random.randint(2*self.width//3,self.width),np.random.randint(self.height)))
        random_pos_zone3 = list(random_pos_zone3)


        for w in range(self.num_waste):
            agent = WasteAgent(self,color='green')
            pos = random_pos_zone1.pop()
            self.grid.place_agent(agent,pos)
            self.waste_agents.append(agent)
            

        for g in range(self.num_green):
            agent = RobotAgent(self,color='green')
            pos = random_pos_zone1.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)
        for y in range(self.num_yellow):
            agent = RobotAgent(self,color='yellow')
            pos = random_pos_zone2.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)
        for r in range(self.num_red):
            agent = RobotAgent(self,color='red')
            pos = random_pos_zone3.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)


    def step_agents(self):
        shuffled = np.random.permutation(self.robot_agents)
        for agent in shuffled:
            #find agent position and find the neighbours
            neighbour_squares = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=True
            )
            curr_x,curr_y = agent.pos
            observation = {}
            for (x,y) in neighbour_squares:
                #get all agents on that square
                cellmates = self.grid.get_cell_list_contents([(x,y)])
                observation[(x-curr_x,y-curr_y)] = cellmates
            #to change
            agent.step_agent(observation)

        self.datacollector.collect(self)

    def move(self,agent,direction):
        new_agent_pos = (agent.pos[0] + direction_dict[direction][0],  agent.pos[1] + direction_dict[direction][1])
        self.grid.move_agent(agent,new_agent_pos)
        waste_ids = agent.knowledge['transporting']
        for _id in waste_ids:
            self.grid.move_agent(self.get_agent_by_id(_id),new_agent_pos)

    def get_agent_by_id(self,id):
        for agent in self.agents:
            if id == agent.unique_id:
                return agent

    def do(self,agent,action):
        '''
        perform action
        action can be either 'move' or 'drop'
        when dropping, you do not move.
        '''
        if not isinstance(action,NoneAction): # if action is nothing, do nothing ( useful for radioactivity agents and waste agents)

            #get the current position of the agent
            if isinstance(action,Drop):
                #get all
                agent.drop(action.drop_id)
                dropped = self.get_agent_by_id(action.drop_id)
                dropped.picked_up = False

                #check if its arrived
                if dropped.color == "red" and dropped.pos == (self.width -1,self.height-1):
                    dropped.arrived = True
                    self.disposed_waste_count += 1
                #print(f"Dropping waste {action.drop_id}")

            if isinstance(action,Move):
                self.move(agent,action.direction)
                #print(f"currently transporting {agent.knowledge['transporting']}")
                #print(f"moving in direction {action.direction}")
            #if can pickup, do it
            cellmates = self.grid.get_cell_list_contents([agent.pos])
            for cellmate in cellmates:
                if isinstance(cellmate,WasteAgent) and cellmate.color == agent.color and not cellmate.picked_up\
                and not cellmate.arrived and len(agent.knowledge['transporting'])<=1:
                    agent.pickup(cellmate.unique_id)
                    cellmate.picked_up = True
                    print(f"picking up {cellmate.unique_id} of color {cellmate.color}")

            #transform wastes when two in the same bag
            if len(agent.knowledge['transporting']) == 2:
                #print(agent.knowledge['transporting'])
                id1,id2 = agent.knowledge['transporting'][0],agent.knowledge['transporting'][1]
                if self.get_agent_by_id(id1).color == 'red' or self.get_agent_by_id(id1).color != self.get_agent_by_id(id2).color :
                    pass
                else:
                    #print("changing color of agent", self.get_agent_by_id(id1).unique_id,id1)
                    self.get_agent_by_id(id1).color = next_color(self.get_agent_by_id(id1).color)
                    #print('removing agent', id2)
                    self.grid.remove_agent(self.get_agent_by_id(id2))
                    self.agents.remove(self.get_agent_by_id(id2))
                    agent.knowledge['transporting'].pop()


    def step(self):
        if self.running:
            self.step_agents()
            self.datacollector.collect(self)
        else:
            pass # Model is paused, do nothing