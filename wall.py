class Wall:
    positions = []
    
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        Wall.positions.append((x,y))
        
    def __str__(self) -> str:
        return "#"