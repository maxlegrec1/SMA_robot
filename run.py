# run.py
import model
import matplotlib.pyplot as plt
from server import MatplotlibVisualization
import time
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
# Create the model (adjust parameters as needed)
mod = model.WasteRetrievalModel(num_green = 2, num_yellow = 2, num_red = 1, num_waste = 5, width = 30, height = 30)  # Using smaller grid for better visualization
num_steps = 1000

# Create the visualization
viz = MatplotlibVisualization(mod)
plt.show(block=False)

# Start interactive loop - simulation is controlled by play/pause button in visualization now
while True:
    #print(viz.model.running)
    if viz.model.running:
        viz.model.step() # Step the model only if running is True
        viz.render()
    else:
        viz.render() # Still render to update display even when paused
        #plt.pause(0.001) # Small pause to keep UI responsive, reduce if needed