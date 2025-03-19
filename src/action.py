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
class Action():
    def __init__(self):
        pass
class Move(Action):
    def __init__(self,direction):
        self.direction = direction
class Drop(Action):
    def __init__(self,drop_id):
        self.drop_id = drop_id

class NoneAction(Action):
    def __init__(self):
        pass