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
import pandas as pd
import numpy as np
import os
from copy import copy
import json
from .agents import RadioactivityAgent, RobotAgent, WasteAgent, RefinedAgent
from .action import Move, Drop, NoneAction
from .variables import color_dict,direction_dict,inv_direction_dict,robot_dict

def next_color(color):
    if color=='green':
        return 'yellow'
    if color=="yellow":
        return "red"
    else:
        raise Exception(f"Invalid color {color} for next color")

class WasteRetrievalModel(Model):
    def __init__(self,
                 num_green = 1,
                 num_yellow =1,
                 num_red = 1,
                 num_waste_yellow = 0,
                 num_waste_red = 0,
                 num_waste_green = 4,
                 width = 30,
                 height = 30,
                 seed=None,
                 strategy='refined',
                 save_path = "results/",
                 max_steps = 20000,
                 finish_threshold = 0.9):
        super().__init__(seed=seed)
        if(width%3!=0):
            raise Exception("The indicated width is not a multiple of 3")
        self.save_path = save_path
        self.max_steps = max_steps
        self.finish_threshold = finish_threshold
        self.width = width
        self.height = height
        self.num_green = num_green
        self.num_yellow = num_yellow
        self.num_red = num_red
        self.num_waste_green = num_waste_green
        self.num_waste_yellow = num_waste_yellow
        self.num_waste_red = num_waste_red
        self.potential_red =  num_waste_green // 4 + num_waste_yellow // 2 + num_waste_red
        self.current_step = 0
        self.finished = False
        self.grid = mesa.space.MultiGrid(width, height,torus = False)
        if strategy == 'communication':
            self.communicate = True
            self.strategy = 'refined'
        else:
            self.communicate = False
            self.strategy = strategy
    
        

        #self.schedule = mesa.time.RandomActivation(self)
        self.disposed_waste_count = 0
        # Add the datacollector
        self.datacollector = DataCollector(
            model_reporters={
                "Green Waste": self.count_green_waste,
                "Yellow Waste": self.count_yellow_waste,
                "Red Waste": self.count_red_waste,
                "Disposed Waste": self.count_disposed_waste,
                "Progress": self.calculate_progress,
            },
            agenttype_reporters=
            {
                RadioactivityAgent: {"Radioactivity": lambda a: a.radioactivity},

                RobotAgent: {"Color": lambda a: a.color,
                            "Transporting": lambda a: copy(a.knowledge['transporting']),#beware of lists :)
                            "Position": lambda a: a.pos},

                WasteAgent: {"Color": lambda a: a.color,
                            "Picked Up": lambda a: a.picked_up,
                            "Arrived": lambda a: a.arrived}
            }
        )
        self.running = True # Simulation starts paused

        self.robot_agents = []
        self.waste_agents = []
        self.radioactivity_agents = []
        self.initialize_agents()
        self.datacollector.collect(self)
        self.create_config()
    def initialize_agents(self):
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.robot_agents = []
        self.waste_agents = []
        self.radioactivity_agents = []
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
                        agent = RadioactivityAgent(self,radioactivity=zone/3 + np.random.random()/3)
                        self.grid.place_agent(agent, (i, j))
                        self.radioactivity_agents.append(agent)


        random_pos_zone1 = set()
        while len(random_pos_zone1) < self.num_waste_green + self.num_green:
            random_pos_zone1.add((np.random.randint(self.width//3),np.random.randint(self.height)))
        random_pos_zone1 = list(random_pos_zone1)

        random_pos_zone2 = set()
        while len(random_pos_zone2) < self.num_waste_yellow + self.num_yellow:
            random_pos_zone2.add((np.random.randint(self.width//3,2*self.width//3),np.random.randint(self.height)))
        random_pos_zone2 = list(random_pos_zone2)

        random_pos_zone3 = set()
        while len(random_pos_zone3) < self.num_waste_red + self.num_red:
            random_pos_zone3.add((np.random.randint(2*self.width//3,self.width),np.random.randint(self.height)))
        random_pos_zone3 = list(random_pos_zone3)


        for w in range(self.num_waste_green):
            agent = WasteAgent(self,color='green')
            pos = random_pos_zone1.pop()
            self.grid.place_agent(agent,pos)
            self.waste_agents.append(agent)
        for w in range(self.num_waste_yellow):
            agent = WasteAgent(self,color='yellow')
            pos = random_pos_zone2.pop()
            self.grid.place_agent(agent,pos)
            self.waste_agents.append(agent)
        for w in range(self.num_waste_red):
            agent = WasteAgent(self,color='red')
            pos = random_pos_zone3.pop()
            self.grid.place_agent(agent,pos)
            self.waste_agents.append(agent)

        for g in range(self.num_green):
            agent = eval(robot_dict[self.strategy])(self,color='green')
            pos = random_pos_zone1.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)
        for y in range(self.num_yellow):
            agent = eval(robot_dict[self.strategy])(self,color='yellow')
            pos = random_pos_zone2.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)
        for r in range(self.num_red):
            agent = eval(robot_dict[self.strategy])(self,color='red')
            pos = random_pos_zone3.pop()
            self.grid.place_agent(agent,pos)
            self.robot_agents.append(agent)


    def step_agents(self):
        shuffled = np.random.permutation(self.robot_agents)
        ### Communication ###
        if self.communicate:
            seen_pairs = set()
            agent_tuples = []

            for agent1 in self.robot_agents:
                x1, y1 = agent1.pos
                neighbors = self.grid.get_neighborhood((x1, y1), moore=True, include_center=False)

                for (nx, ny) in neighbors:
                    x_offset = nx - x1
                    y_offset = ny - y1

                    cellmates = self.grid.get_cell_list_contents([(nx, ny)])
                    for agent2 in cellmates:
                        if isinstance(agent2, type(agent1)):
                            pair_id = tuple(sorted((id(agent1), id(agent2))))
                            if pair_id in seen_pairs:
                                continue
                            seen_pairs.add(pair_id)

                            agent2_x = agent2.knowledge["agent_x"]
                            agent2_y = agent2.knowledge["agent_y"]
                            agent1_in_agent2 = (agent2_x - x_offset, agent2_y - y_offset)

                            agent_tuples.append((agent1, agent2, agent1_in_agent2))

            for (agent1, agent2, agent1_in_agent2) in agent_tuples:
                merged, agent1_in_merged, agent2_in_merged = self.merge_agents_knowledge(
                    agent1=agent1,
                    agent2=agent2,
                    agent1_in_agent2=agent1_in_agent2,
                )
                agent1.knowledge['internal_map'] = merged
                agent2.knowledge['internal_map'] = merged

                agent1.knowledge['agent_x'] = agent1_in_merged[0]
                agent2.knowledge['agent_x'] = agent2_in_merged[0]

                agent1.knowledge['agent_y'] = agent1_in_merged[1]
                agent2.knowledge['agent_y'] = agent2_in_merged[1]
                

        for agent in shuffled:
            #find agent position and find the neighbours
            neighbour_squares = self.grid.get_neighborhood(
            agent.pos, moore=True, include_center=True
            )
            curr_x,curr_y = agent.pos

            # If the agent share this cell with another agent of the same color
            # and they hold the exactly one of the same type of waste 
            if len(agent.knowledge['transporting']) == 1 and agent.state == "FINDING_WASTE":
                # if the agent is transporting one and if it is still looking then we know
                # it is tranporting a waste of a color below the color of the agent.
                cellmates = self.grid.get_cell_list_contents([(curr_x, curr_y)])
                for cellmate in cellmates: 
                    if isinstance(cellmate, type(agent)):
                        if (cellmate.color == agent.color) and (cellmate.state == "FINDING_WASTE"):
                            if len(cellmate.knowledge['transporting']) == 1:
                                transported_waste_id = cellmate.knowledge['transporting'][0]
                                cellmate.drop(transported_waste_id)
                                agent.pickup(transported_waste_id)
            observation = {}
            for (x,y) in neighbour_squares:
                #get all agents on that square
                cellmates = self.grid.get_cell_list_contents([(x,y)])
                observation[(x-curr_x,y-curr_y)] = cellmates
            #to change
            agent.step_agent(observation)

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
        if self.running and not self.finished:
            self.step_agents()
            self.current_step += 1
            self.datacollector.collect(self)
            self.check_finished()
        else:
            pass # Model is paused, do nothing
        
    import numpy as np

    def merge_agents_knowledge(self,agent1, agent2, agent1_in_agent2, fill_value=-1):
        map1 = agent1.knowledge_map
        map2 = agent2.knowledge_map
        h1, w1, c = map1.shape
        h2, w2, _ = map2.shape

        assert c == 6, "Each map is expected to have 6 channels"

        ref1 = (agent1.knowledge['agent_y'], agent1.knowledge['agent_x'])  # (row, col)
        agent2_pos = (agent2.knowledge['agent_y'], agent2.knowledge['agent_x'])
        ref2 = (agent1_in_agent2[1], agent1_in_agent2[0])  # (row, col)

        dy = ref1[0] - ref2[0]
        dx = ref1[1] - ref2[1]

        top = min(0, dy)
        left = min(0, dx)
        bottom = max(h1, dy + h2)
        right = max(w1, dx + w2)
        H = bottom - top
        W = right - left

        # Initialize merged map: zeros for first 5 channels, fill_value (-1) for last
        merged = np.zeros((H, W, c), dtype=map1.dtype)
        merged[..., 5] = fill_value

        merged_age = np.full((H, W), np.inf)

        y1_offset = -top
        x1_offset = -left
        y2_offset = y1_offset + dy
        x2_offset = x1_offset + dx

        # Insert map1
        m1_slice = merged[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1]
        m1_age = map1[..., 5]
        mask1 = m1_age != -1
        m1_slice[mask1] = map1[mask1]
        merged_age[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1][mask1] = m1_age[mask1]

        # Insert newer data from map2
        m2_age = map2[..., 5]
        y2s, x2s = np.meshgrid(np.arange(h2), np.arange(w2), indexing="ij")
        global_y = y2s + y2_offset
        global_x = x2s + x2_offset
        known_mask = m2_age != -1
        newer_mask = known_mask & (m2_age < merged_age[global_y, global_x])
        merged[global_y[newer_mask], global_x[newer_mask]] = map2[newer_mask]
        merged_age[global_y[newer_mask], global_x[newer_mask]] = m2_age[newer_mask]

        agent1_in_merged = (ref1[1] + x1_offset, ref1[0] + y1_offset)
        agent2_in_merged = (agent2_pos[1] + x2_offset, agent2_pos[0] + y2_offset)

        return merged, agent1_in_merged, agent2_in_merged


    def count_green_waste(self):
        return sum(1 for agent in self.agents if isinstance(agent, WasteAgent) and agent.color == 'green')

    def count_yellow_waste(self):
        return sum(1 for agent in self.agents if isinstance(agent, WasteAgent) and agent.color == 'yellow')

    def count_red_waste(self):
        return sum(1 for agent in self.agents if isinstance(agent, WasteAgent) and agent.color == 'red')

    def count_disposed_waste(self):
        # Count waste that has been properly disposed (you may need to add a flag to track this)
        return self.disposed_waste_count
    
    def calculate_progress(self):
        return self.disposed_waste_count / self.potential_red
    
    def check_finished(self):
        if self.calculate_progress() >= self.finish_threshold or self.current_step >= self.max_steps:
            self.finished = True
            print(f"Finishing simulation due to progress: {self.calculate_progress()} or step: {self.current_step}")
            self.running = False
            self.save_data()
    
    def save_data(self):
        # Save the data to a CSV file
        num_runs = len(os.listdir(self.save_path))
        run_dir = self.save_path + f"simulation_{num_runs}"
        os.makedirs(run_dir, exist_ok=True)
        with open(f"{run_dir}/config.json", "w") as f:
            json.dump(self.config, f)
        model_data = self.datacollector.get_model_vars_dataframe()
        model_data.to_csv(f"{run_dir}/model.csv")
        data_radioactivity = self.datacollector.get_agenttype_vars_dataframe(RadioactivityAgent)
        data_radioactivity.to_csv(f"{run_dir}/agent_radioactivity.csv")
        data_waste = self.datacollector.get_agenttype_vars_dataframe(WasteAgent)
        data_waste.to_csv(f"{run_dir}/agent_waste.csv")
        data_robot = self.datacollector.get_agenttype_vars_dataframe(RobotAgent)
        data_robot.to_csv(f"{run_dir}/agent_robot.csv")

        print(f"Data saved to {run_dir}")

    def create_config(self):
        self.config = {
            "num_green": self.num_green,
            "num_yellow": self.num_yellow,
            "num_red": self.num_red,
            "num_waste_green": self.num_waste_green,
            "num_waste_yellow": self.num_waste_yellow,
            "num_waste_red": self.num_waste_red,
            "width": self.width,
            "height": self.height,
            "save_path": self.save_path,
            "max_steps": self.max_steps,
            "finish_threshold": self.finish_threshold
        }