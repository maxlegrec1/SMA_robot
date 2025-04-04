from src.agents import RadioactivityAgent, RobotAgent, WasteAgent
from src.model import WasteRetrievalModel

from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)

radioactivity_colors = ["green", "gold", "red"]
robot_colors = {"green":"lawngreen",
                "yellow":"yellow",
                "red":"darkred"}
waste_colors = {"green":"springgreen",
                "yellow":"darkkhaki",
                "red":"indianred"}

def agent_portrayal(agent):

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, RadioactivityAgent):
        portrayal["marker"] = "s"
        portrayal["zorder"] = 0
        portrayal["size"] = 100
        if agent.radioactivity != 2:
            portrayal["color"] = radioactivity_colors[int(3*agent.radioactivity)]
        else:
            portrayal["color"] = radioactivity_colors[agent.radioactivity]

    elif isinstance(agent, WasteAgent):
        portrayal["color"] = waste_colors[agent.color]
        portrayal["marker"] = "v"
        portrayal["zorder"] = 2
        
    elif isinstance(agent, RobotAgent):
        portrayal["color"] = robot_colors[agent.color]
        portrayal["marker"] = "s"
        portrayal["size"] = 75

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "num_green": Slider("Number of green robots", 5, 1, 10),
    "num_yellow": Slider("Number of yellow robots", 5, 1, 10),
    "num_red": Slider("Number of red robots", 5, 1, 10),
    "num_waste": Slider("Number of initial waste units", 10, 1, 30),
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

lineplot_component = make_plot_component(
    {"Green Waste": "green", "Red Waste": "red", "Yellow Waste": "yellow", "Disposed Waste":"black"},
    post_process=post_process_lines,
)

# simulator = ABMSimulator()
model = WasteRetrievalModel()
# model.running = True

page = SolaraViz(
    model,
    components=[space_component, lineplot_component],
    model_params=model_params,
    name="Radioactive waste retrieval",
    # simulator=simulator,
)
page  # noqa
