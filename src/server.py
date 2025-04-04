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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
import matplotlib.transforms as transforms

from .agents import RadioactivityAgent, RobotAgent, WasteAgent
class MatplotlibVisualization:
    """An interactive visualization class using Matplotlib for the Waste Retrieval Model"""

    def __init__(self, model):
        self.model = model
        # Create a figure with a larger right margin for statistics and controls
        self.fig, self.ax = plt.subplots(figsize=(16, 10)) # Increased figure width
        # Adjust layout to make room for the legend, stats, and controls on the right
        self.fig.subplots_adjust(right=0.65, left=0.05) # Adjusted right and left

        plt.ion()  # Turn on interactive mode

        # Define the sub-positions within each cell (3x3 grid)
        offset = 0.2
        self.sub_positions = [
            (-offset, -offset), (0, -offset), (offset, -offset),
            (-offset, 0), (0, 0), (offset, 0),
            (-offset, offset), (0, offset), (offset, offset)
        ]

        # Enable zooming and panning
        plt.rcParams['toolbar'] = 'toolmanager'  # Use the toolmanager backend
        self.ax.set_aspect('equal')  # Preserve aspect ratio when zooming

        # Enable zoom and pan by default
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)  # Zoom with scroll wheel
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        # Setup for pan functionality
        self._pan_start = None

        # Setup for tooltips
        self.tooltip_text = None
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)

        # Add status text area for hover information
        self.status_text = self.fig.text(0.02, 0.02, "", bbox=dict(facecolor='white', alpha=0.7))

        # Store scatter plot objects for updating
        self.scatters = []

        # Store positions and agent mappings for hover detection
        self.agent_positions = {}  # (x, y) -> agent mapping

        # Store current view limits to preserve zoom
        self.current_xlim = None
        self.current_ylim = None

        # Store text elements for updating
        self.nav_text = None
        self.stats_text = None
        self.legend = None

        # Initial background grid drawn once
        self._draw_background()
        self._add_controls() # Add control buttons and sliders

        # Initialize params with initial slider values - FIX for AttributeError
        self.params = {
            'num_green': self.slider_num_green.val,
            'num_yellow': self.slider_num_yellow.val,
            'num_red': self.slider_num_red.val,
            'num_waste_green': self.slider_num_waste_green.val,
            'num_waste_yellow': self.slider_num_waste_yellow.val,
            'num_waste_red': self.slider_num_waste_red.val,
            'width': self.slider_width.val,
            'height': self.slider_height.val
        }


    def _add_controls(self):
        """Add control buttons and sliders to the visualization."""
        # Button and slider positions (adjust as needed)
        button_ax_play_pause = plt.axes([0.70, 0.85, 0.1, 0.04]) # [left, bottom, width, height]
        button_ax_reset = plt.axes([0.70, 0.78, 0.1, 0.04])

        # Slider positions
        slider_ax_num_green = plt.axes([0.70, 0.65, 0.25, 0.03])
        slider_ax_num_yellow = plt.axes([0.70, 0.60, 0.25, 0.03])
        slider_ax_num_red = plt.axes([0.70, 0.55, 0.25, 0.03])
        slider_ax_num_waste_green = plt.axes([0.70, 0.50, 0.25, 0.03])
        slider_ax_num_waste_yellow = plt.axes([0.70, 0.45, 0.25, 0.03])
        slider_ax_num_waste_red = plt.axes([0.70, 0.40, 0.25, 0.03])
        slider_ax_width = plt.axes([0.70, 0.35, 0.25, 0.03])
        slider_ax_height = plt.axes([0.70, 0.30, 0.25, 0.03])

        # Buttons
        self.button_play_pause = Button(button_ax_play_pause, 'Play')
        self.button_play_pause.on_clicked(self.toggle_play_pause)
        self.button_reset = Button(button_ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_simulation)

        # Sliders
        self.slider_num_green = Slider(slider_ax_num_green, 'Green Robots', 0, 10, valinit=self.model.num_green, valstep=1)
        self.slider_num_yellow = Slider(slider_ax_num_yellow, 'Yellow Robots', 0, 10, valinit=self.model.num_yellow, valstep=1)
        self.slider_num_red = Slider(slider_ax_num_red, 'Red Robots', 0, 10, valinit=self.model.num_red, valstep=1)
        self.slider_num_waste_green = Slider(slider_ax_num_waste_green, 'Green Waste', 0, 20, valinit=self.model.num_waste_green, valstep=1)
        self.slider_num_waste_yellow = Slider(slider_ax_num_waste_yellow, 'Yellow Waste', 0, 20, valinit=self.model.num_waste_yellow, valstep=1)
        self.slider_num_waste_red = Slider(slider_ax_num_waste_red, 'Red Waste', 0, 20, valinit=self.model.num_waste_red, valstep=1)
        self.slider_width = Slider(slider_ax_width, 'Width', 30, 90, valinit=self.model.width, valstep=3) # Width must be multiple of 3
        self.slider_height = Slider(slider_ax_height, 'Height', 30, 90, valinit=self.model.height, valstep=1)

        # Connect sliders to update function
        self.slider_num_green.on_changed(self.update_params)
        self.slider_num_yellow.on_changed(self.update_params)
        self.slider_num_red.on_changed(self.update_params)
        self.slider_num_waste_green.on_changed(self.update_params)
        self.slider_num_waste_yellow.on_changed(self.update_params)
        self.slider_num_waste_red.on_changed(self.update_params)
        self.slider_width.on_changed(self.update_params)
        self.slider_height.on_changed(self.update_params)

    def update_params(self, val):
        """Update model parameters based on slider values."""
        self.params = {
            'num_green': int(self.slider_num_green.val),
            'num_yellow': int(self.slider_num_yellow.val),
            'num_red': int(self.slider_num_red.val),
            'num_waste_green': int(self.slider_num_waste_green.val),
            'num_waste_yellow': int(self.slider_num_waste_yellow.val),
            'num_waste_red': int(self.slider_num_waste_red.val),
            'width': int(self.slider_width.val),
            'height': int(self.slider_height.val)
        }

    def toggle_play_pause(self, event):
        """Toggle play/pause state of the simulation."""
        self.model.running = not self.model.running
        if self.model.running:
            self.button_play_pause.label.set_text('Pause')
        else:
            self.button_play_pause.label.set_text('Play')
        plt.draw()

    def reset_simulation(self, event):
        """Reset the simulation with current parameters."""
        params = self.params
        try:
            new_model = type(self.model)(**params) # Re-initialize the model with current params
            self.model = new_model
            self.button_play_pause.label.set_text('Play') # Reset button to 'Play' state
            self.model.running = False # Ensure it starts paused
            self._draw_background() # Redraw background in case width/height changed
            self.current_xlim = self.default_xlim # Reset zoom
            self.current_ylim = self.default_ylim # Reset zoom
            self.ax.set_xlim(self.default_xlim) # Apply reset zoom
            self.ax.set_ylim(self.default_ylim) # Apply reset zoom
        except Exception as e:
            print(f"Error resetting simulation: {e}")
            plt.title(f'Error: {e}', color='red') # Display error on plot
        self.render() # Initial render after reset

    def _draw_background(self):
        """Draw the static background elements of the visualization"""
        # Clear previous background if exists (for reset)
        if hasattr(self, 'background_img'):
            self.background_img.remove()

        # Create a grid representation
        grid = np.zeros((self.model.grid.height, self.model.grid.width, 3))

        # Color the zones
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                if x < self.model.grid.width // 3:
                    grid[y, x] = [0.8, 1.0, 0.8]  # Light green for zone 1
                elif x < 2 * self.model.grid.width // 3:
                    grid[y, x] = [1.0, 1.0, 0.8]  # Light yellow for zone 2
                else:
                    grid[y, x] = [1.0, 0.8, 0.8]  # Light red for zone 3

        # Draw the grid
        self.background_img = self.ax.imshow(grid, origin='lower')

        # Set grid lines
        self.ax.set_xticks(np.arange(-0.5, self.model.grid.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.model.grid.height, 1), minor=True)
        self.ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.2)

        # Draw zone borders
        zone1_border = self.model.grid.width // 3 - 0.5
        zone2_border = 2 * self.model.grid.width // 3 - 0.5
        self.ax.axvline(x=zone1_border, color='black', linestyle='--', linewidth=2)
        self.ax.axvline(x=zone2_border, color='black', linestyle='--', linewidth=2)

        # Add zone labels above the grid
        self.ax.text(self.model.grid.width // 6, self.model.grid.height + 0.5, 'Zone 1\nLow Radioactivity',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        self.ax.text(self.model.grid.width // 2, self.model.grid.height + 0.5, 'Zone 2\nMedium Radioactivity',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        self.ax.text(5 * self.model.grid.width // 6, self.model.grid.height + 0.5, 'Zone 3\nHigh Radioactivity',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        # Set default view limits
        padding = 0.1
        self.default_xlim = (-0.5 + padding, self.model.grid.width - 0.5 - padding)
        self.default_ylim = (-0.5 + padding, self.model.grid.height + 2 - padding)
        self.ax.set_xlim(self.default_xlim)
        self.ax.set_ylim(self.default_ylim)

        # Set title
        self.ax.set_title('Waste Retrieval Simulation')

        # Add navigation instructions
        nav_text = (
            "Navigation:\n"
            "• Scroll to zoom in/out\n"
            "• Left-click and drag to pan\n"
            "• Hover over agents for info"
        )
        if self.nav_text:
            self.nav_text.remove() # Remove old text if it exists (for reset)
        self.nav_text = plt.figtext(0.02, 0.95, nav_text, fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')


    def _on_scroll(self, event):
        """Handle scroll events for zooming"""
        if event.inaxes != self.ax:
            return

        # Get current axis limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        # Calculate zoom factor
        base_scale = 1.1
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        else:
            # Zoom out
            scale_factor = base_scale

        # Get the coordinates of the mouse position
        x_coord = event.xdata
        y_coord = event.ydata

        # Calculate new axis limits while keeping the mouse position fixed
        new_width = (x_max - x_min) * scale_factor
        new_height = (y_max - y_min) * scale_factor

        # Calculate the new limits
        x_min_new = x_coord - (x_coord - x_min) * scale_factor
        x_max_new = x_coord + (x_max - x_coord) * scale_factor
        y_min_new = y_coord - (y_coord - y_min) * scale_factor
        y_max_new = y_coord + (y_max - y_coord) * scale_factor

        # Set the new limits
        self.ax.set_xlim(x_min_new, x_max_new)
        self.ax.set_ylim(y_min_new, y_max_new)

        # Store current view limits
        self.current_xlim = (x_min_new, x_max_new)
        self.current_ylim = (y_min_new, y_max_new)

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        """Handle mouse button press for panning"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left mouse button
            self._pan_start = (event.xdata, event.ydata)
            try:
                self.fig.canvas.get_tk_widget().config(cursor="fleur")  # Change cursor to panning
            except:
                pass  # If not using tkinter backend

    def _on_release(self, event):
        """Handle mouse button release to end panning"""
        if event.button == 1:  # Left mouse button
            self._pan_start = None
            try:
                self.fig.canvas.get_tk_widget().config(cursor="")  # Reset cursor
            except:
                pass  # If not using tkinter backend

    def _on_motion(self, event):
        """Handle mouse motion for panning"""
        if self._pan_start is None or event.inaxes != self.ax:
            return

        # Calculate how much the mouse has moved
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]

        # Get current axis limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        # Set new limits
        self.ax.set_xlim(x_min - dx, x_max - dx)
        self.ax.set_ylim(y_min - dy, y_max - dy)

        # Store current view limits
        self.current_xlim = (x_min - dx, x_max - dx)
        self.current_ylim = (y_min - dy, y_max - dy)

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def _on_hover(self, event):
        """Handle hover events to display tooltips"""
        if event.inaxes != self.ax:
            self.status_text.set_text("")
            return

        # Get the cursor position
        cursor_x, cursor_y = event.xdata, event.ydata

        # Find the closest agent
        closest_agent = None
        min_distance = 0.3  # Threshold for detection

        for pos, agent in self.agent_positions.items():
            # Calculate distance to mouse position
            distance = np.sqrt((pos[0] - cursor_x)**2 + (pos[1] - cursor_y)**2)

            if distance < min_distance:
                min_distance = distance
                closest_agent = agent

        if closest_agent:
            if isinstance(closest_agent, WasteAgent):
                info = f"Waste ID: {closest_agent.unique_id}, Type: {closest_agent.color.capitalize()}"
                if closest_agent.picked_up:
                    info += f", Status: Being transported"
            elif isinstance(closest_agent, RobotAgent):
                info = f"Robot ID: {closest_agent.unique_id}, Type: {closest_agent.color.capitalize()}"
                if closest_agent.knowledge['transporting']:
                    waste_ids = closest_agent.knowledge['transporting']
                    carried_waste = []
                    for waste_id in waste_ids:
                        for model_agent in self.model.agents:
                            if isinstance(model_agent, WasteAgent) and model_agent.unique_id == waste_id:
                                carried_waste.append(f"{model_agent.color.capitalize()} Waste (ID: {waste_id})")

                    if carried_waste:
                        info += f"\nCarrying: {', '.join(carried_waste)}"
            else:
                info = f"Agent ID: {closest_agent.unique_id}"

            self.status_text.set_text(info)
        else:
            self.status_text.set_text("")

    def render(self):
        """Render the grid and agents while preserving zoom level"""
        # Store current view limits before clearing
        if self.current_xlim is None:
            self.current_xlim = self.ax.get_xlim()
        if self.current_ylim is None:
            self.current_ylim = self.ax.get_ylim()

        # Remove previous agent scatters
        for scatter in self.scatters:
            scatter.remove()
        self.scatters = []

        # Clear agent positions mapping
        self.agent_positions = {}

        # Remove previous legend and stats if they exist
        if self.legend:
            self.legend.remove()
        if self.stats_text:
            self.stats_text.remove()

        # Collect all agents by cell for strategic positioning
        cell_agents = {}
        for cell_content, (x, y) in self.model.grid.coord_iter():
            if cell_content:  # If there are agents in this cell
                cell_agents[(x, y)] = []
                for agent in cell_content:
                    if not isinstance(agent, RadioactivityAgent):  # Skip radioactivity agents
                        cell_agents[(x, y)].append(agent)

        # Draw agents with sub-positions
        for (x, y), agents in cell_agents.items():
            # Skip if no visible agents
            if not agents:
                continue

            # Distribute agents across sub-positions
            for i, agent in enumerate(agents[:9]):  # Limit to 9 agents per cell
                # Calculate sub-position
                dx, dy = self.sub_positions[i]
                sub_x, sub_y = x + dx, y + dy

                if isinstance(agent, WasteAgent):
                    marker = 'D'  # Diamond for waste
                    size = 40
                    if agent.color == "green":
                        color = "green"
                    elif agent.color == "yellow":
                        color = "yellow"
                    else:  # red
                        color = "red"

                    # If picked up, don't render here (it will be shown on top of robot)
                    if not agent.picked_up:
                        scatter = self.ax.scatter(sub_x, sub_y, c=color, marker=marker, s=size, alpha=1.0,
                                                 zorder=10)
                        self.scatters.append(scatter)
                        # Store position for hover detection
                        self.agent_positions[(sub_x, sub_y)] = agent

                elif isinstance(agent, RobotAgent):
                    marker = 'o'  # Circle for robots
                    if agent.color == "green":
                        color = "darkgreen"
                    elif agent.color == "yellow":
                        color = "olive"
                    else:  # red
                        color = "darkred"

                    # Draw the robot
                    scatter = self.ax.scatter(sub_x, sub_y, c=color, marker=marker, s=90,
                                             zorder=5)
                    self.scatters.append(scatter)
                    # Store position for hover detection
                    self.agent_positions[(sub_x, sub_y)] = agent

                    # If carrying waste, draw waste on top
                    if agent.knowledge['transporting']:
                        # Get the first waste being transported
                        waste_color = None
                        waste_agent = None
                        for waste_id in agent.knowledge['transporting']:
                            # Find the waste in the model's agent list
                            for model_agent in self.model.agents:
                                if isinstance(model_agent, WasteAgent) and model_agent.unique_id == waste_id:
                                    waste_color = model_agent.color
                                    waste_agent = model_agent
                                    break
                            if waste_color:
                                break

                        if waste_color:
                            # Draw smaller waste on top of robot
                            if waste_color == "green":
                                w_color = "green"
                            elif waste_color == "yellow":
                                w_color = "yellow"
                            else:  # red
                                w_color = "red"

                            # Draw waste slightly above robot
                            scatter = self.ax.scatter(sub_x, sub_y + 0.1, c=w_color, marker='D', s=25, alpha=0.8,
                                                    zorder=15)
                            self.scatters.append(scatter)
                            if waste_agent:
                                # Store position for hover detection
                                self.agent_positions[(sub_x, sub_y + 0.1)] = waste_agent

        # Add legend to the right of the grid
        '''
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=12, label='Green Robot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='olive', markersize=12, label='Yellow Robot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=12, label='Red Robot'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, label='Green Waste'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', markersize=8, label='Yellow Waste'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8, label='Red Waste'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=12, label='Robot with Waste',
                      markeredgecolor='darkgreen'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=6, alpha=0.8, label='Carried Waste')
        ]'''

        # Position the legend outside the grid on the right side
        #self.legend = self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Count agents by type for statistics
        green_waste = sum(1 for agent in self.model.agents if isinstance(agent, WasteAgent) and agent.color == 'green')
        yellow_waste = sum(1 for agent in self.model.agents if isinstance(agent, WasteAgent) and agent.color == 'yellow')
        red_waste = sum(1 for agent in self.model.agents if isinstance(agent, WasteAgent) and agent.color == 'red')
        disposed_waste = self.model.disposed_waste_count

        # Count carried waste
        carried_waste = sum(1 for agent in self.model.agents if isinstance(agent, WasteAgent) and agent.picked_up)

        # Count robots by type
        green_robots = sum(1 for agent in self.model.agents if isinstance(agent, RobotAgent) and agent.color == 'green')
        yellow_robots = sum(1 for agent in self.model.agents if isinstance(agent, RobotAgent) and agent.color == 'yellow')
        red_robots = sum(1 for agent in self.model.agents if isinstance(agent, RobotAgent) and agent.color == 'red')

        # Display statistics in a text box on the right side of the figure
        stats_text = (
            "STATISTICS\n\n"
            f"Green Waste: {green_waste}\n"
            f"Yellow Waste: {yellow_waste}\n"
            f"Red Waste: {red_waste}\n"
            f"Disposed Waste: {disposed_waste}\n"
            f"Total Waste: {green_waste + yellow_waste + red_waste}\n\n"
            f"Carried Waste: {carried_waste}\n\n"
            f"Green Robots: {green_robots}\n"
            f"Yellow Robots: {yellow_robots}\n"
            f"Red Robots: {red_robots}\n"
            f"Total Robots: {green_robots + yellow_robots + red_robots}"
        )

        # Add a text box for statistics to the right of the grid
        '''
        if self.stats_text:
            self.stats_text.remove() # Remove old stats text if it exists (for update)'''
        self.stats_text = plt.figtext(0.70, 0.05, stats_text, fontsize=10,
                   bbox=dict(facecolor='lightgray', alpha=0.5), verticalalignment='bottom')

        # Restore the view limits to maintain zoom level
        self.ax.set_xlim(self.current_xlim)
        self.ax.set_ylim(self.current_ylim)

        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # Reduced pause for smoother interaction