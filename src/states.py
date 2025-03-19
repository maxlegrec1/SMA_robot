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
STATES = ["FINDING_WASTE","TRANSPORTING","SEARCHING_FOR_BOX"]

#FINDING WASTE = RANDOM WALK IF NOTHING ON INTERNAL MAP, OR GO TO CLOSEST IF SOMETHING ON THE MAP (WE DONT CARE ABOUT AGE YET)

#TRANSPORTING = YOU HAVE TRANSFORMED WASTE AND NOW HAVE TO GO TO THE EAST TO DEPOSE IT ON THE FRONTIER

#SEARCHING FOR BOX , ONLY FOR RED ROBOTS, TO SEE LATER