# SMA_robot/server-mesa.py
from src.agents import RadioactivityAgent, RobotAgent, WasteAgent
from src.model import WasteRetrievalModel

from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)

# Updated color palettes
radioactivity_colors = ['#c6dbef', '#6baed6', '#08306b'] # Light Blue -> Medium Blue -> Dark Blue
robot_colors = {'green': '#98FB98', 'yellow': '#FFD700', 'red': '#DC143C'} # Light Green, Gold, Crimson
waste_colors = {'green': '#90EE90', 'yellow': '#EEE8AA', 'red': '#FFB6C1'} # Pale Green, Pale Goldenrod, Light Pink

def agent_portrayal(agent):

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, RadioactivityAgent):
        portrayal["marker"] = "s"
        portrayal["zorder"] = 0
        portrayal["size"] = 100
        # Assign color based on radioactivity level (assuming levels are 0, 1, 2)
        # Clamp index to avoid errors if radioactivity exceeds expected range
        if agent.radioactivity != 2:
            portrayal["color"] = radioactivity_colors[int(3*agent.radioactivity)]
        else:
            portrayal["color"] = radioactivity_colors[agent.radioactivity]

    elif isinstance(agent, WasteAgent):
        portrayal["color"] = waste_colors[agent.color]
        portrayal["marker"] = "v" # Keep waste as triangles
        portrayal["zorder"] = 2

    elif isinstance(agent, RobotAgent):
        portrayal["color"] = robot_colors[agent.color]
        portrayal["marker"] = "o" # Change robots to circles for better distinction
        portrayal["size"] = 75   # Keep robots slightly smaller than cells
        portrayal["zorder"] = 1  # Ensure robots are drawn above radioactivity layer

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "strategy": {
        "type": "Select",
        "label": "Robot Strategy",
        "value": "random",  # Default strategy
        "values": ["random", "refined", "communication"] # Available strategies
    },
    "num_green": Slider("Number of green robots", 5, 1, 10),
    "num_yellow": Slider("Number of yellow robots", 5, 1, 10),
    "num_red": Slider("Number of red robots", 5, 1, 10),
    "num_waste_green": Slider("Number of initial green waste", 10, 1, 30), # Updated label
    "num_waste_yellow": Slider("Number of initial yellow waste", 10, 1, 30), # Updated label
    "num_waste_red": Slider("Number of initial red waste", 10, 1, 30), # Updated label
    "width": Slider("Grid width", 33, 20, 100),
    "height": Slider("Grid height", 33, 20, 100),
}

def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


space_component = make_space_component(
    agent_portrayal, draw_grid=True, post_process=post_process_space
)

# Updated line plot colors to match waste colors
lineplot_component = make_plot_component(
    {'Green Waste': waste_colors['green'], 'Yellow Waste': waste_colors['yellow'], 'Red Waste': waste_colors['red'], 'Disposed Waste': 'black'},
    post_process=post_process_lines,
)

# simulator = ABMSimulator() # Keep commented unless needed
model = WasteRetrievalModel() # Use the class for SolaraViz instantiation
# model.running = True # Managed by SolaraViz

page = SolaraViz(
    model, # Pass the model class, not an instance
    components=[space_component, lineplot_component],
    model_params=model_params,
    name="Radioactive waste retrieval",
    # simulator=simulator, # Keep commented unless needed
)
page  # noqa E702 This line is needed to render the page in Solara