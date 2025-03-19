

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