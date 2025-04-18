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
from .model import WasteRetrievalModel
import matplotlib.pyplot as plt
from .server import MatplotlibVisualization
import time
# Create the model (adjust parameters as needed)

def Play():
    mod = WasteRetrievalModel(num_green = 5, num_yellow = 5, num_red = 5, num_waste_yellow = 0,num_waste_green=0,num_waste_red=1, width = 30, height = 30, strategy='refined')  # Using smaller grid for better visualization


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