from wall import Wall

class Passage:
    
    def __init__(self, x, y, maze) -> None:
        self.x = x
        self.y = y
        self.maze = maze
        self.connections = self.adjacent_cells(Wall)
        self.adjacent_passages = self.adjacent_cells(Passage)
    
    def adjacent_cells(self, type):
        adj_cells = []
                
        if self.x - 2 >= 0 and isinstance(self.maze[self.x - 2, self.y], type):
            adj_cells.append(self.maze[self.x - 2, self.y])
            
        if self.x + 2 < len(self.maze) and isinstance(self.maze[self.x + 2, self.y], type):
            adj_cells.append(self.maze[self.x + 2, self.y])
            
        if self.y - 2 >= 0 and isinstance(self.maze[self.x, self.y - 2], type):
            adj_cells.append(self.maze[self.x, self.y - 2])
            
        if self.y + 2 < len(self.maze[0]) and isinstance(self.maze[self.x, self.y + 2], type):
            adj_cells.append(self.maze[self.x, self.y + 2])
        
        return adj_cells
    
    def __str__(self):
        return " "