
color_dict = {'green':0,'yellow':1,'red':2}
direction_dict = {"NORTH":(0,1),"SOUTH":(0,-1),"WEST":(-1,0),"EAST":(1,0),"NORTHWEST":(-1,1),"NORTHEAST":(1,1),"SOUTHWEST":(-1,-1),"SOUTHEAST":(1,-1)}
inv_direction_dict = {direction_dict[x]:x for x in direction_dict}
