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
color_dict = {'green':0,'yellow':1,'red':2}
direction_dict = {"NORTH":(0,1),"SOUTH":(0,-1),"WEST":(-1,0),"EAST":(1,0),"NORTHWEST":(-1,1),"NORTHEAST":(1,1),"SOUTHWEST":(-1,-1),"SOUTHEAST":(1,-1)}
inv_direction_dict = {direction_dict[x]:x for x in direction_dict}
robot_dict = {"random":"RobotAgent", "refined":"RefinedAgent"}