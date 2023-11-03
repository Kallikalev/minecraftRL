from wall import Wall
from passage import Passage
import random
import pandas as pd
import numpy as np

class Maze:
    def __init__(self, width: int=7, height: int=7, seed: int=0, loops: int=0) -> None:
        
        self.width = width + 1 if width % 2 == 0 else width
        self.height = height + 1 if height % 2 == 0 else height
        
        self.seed = seed
        self.loops = loops
        
        self.area = np.array([[Wall(x,y) for y in range(self.width)] for x in range(self.height)])
        
        random.seed(self.seed)
        self.spawn_x, self.spawn_y = (random.choice([i for i in range(1, self.height, 2)]), 
                                      random.choice([j for j in range(1, self.width, 2)]))
    
        self.area[self.spawn_x, self.spawn_y] = Passage(self.spawn_x, self.spawn_y, self.area)
        self.pivot_passages = [self.area[self.spawn_x, self.spawn_y]]
        self.frontier_passages = set(self.area[self.spawn_x, self.spawn_y].connections)
        
        self.passages = [Passage(self.spawn_x, self.spawn_y, self.area)]
        
        self.generate()
        self.start = self.random_starting_position()
        self.end = self.random_end_position()
    
    def random_starting_position(self):
        """
        Generates a random start position for a the maze
        """
        
        start = random.choice(self.passages)
        self.area[(start.x, start.y)] = 3
        self.passages.remove(start)
        return (start.x, start.y)
    
    def random_end_position(self):
        """
        Generates a random end position for a the maze
        """
        
        end = random.choice(self.passages)
        self.area[(end.x, end.y)] = 2
        self.passages.remove(end)
        
        return (end.x, end.y)
    
    def find(self, pivot_passages, target_passage):
        """
        Finds any pivot passages that are connected to the target passage
        Returns a random pivot passage that is connected to the target passage
        """
        
        connected_passages = []
        
        for passage in pivot_passages:
            if target_passage in passage.connections:
                connected_passages.append(passage)
                
        return random.choice(connected_passages)
    
    def linked_passage(self, passage_1, passage_2):
        """
        Returns the average position between two passages 
        """
        
        # finds the position between the frontier passage and the pivot passage
        average_position_x = int((passage_1.x + passage_2.x) / 2)
        average_position_y = int((passage_1.y + passage_2.y) / 2)
        
        return (average_position_x, average_position_y)
    
    def generate(self) -> None:
        """
        Generates a maze 
        """
        
        while len(self.frontier_passages) > 0:
            random_pivot_passage = random.choice(sorted(list(self.frontier_passages), key=lambda passage: (passage.x, passage.y)))
            connected_passage = self.find(self.pivot_passages, random_pivot_passage)
            link_passage_x, link_passage_y = self.linked_passage(connected_passage, random_pivot_passage)

            # Connects the Passages together
            self.area[link_passage_x, link_passage_y] = Passage(link_passage_x, link_passage_y, self.area)
            self.passages.append(self.area[link_passage_x, link_passage_y]) # adds the passage into passages list
            connected_passage.connections.remove(random_pivot_passage)
            self.frontier_passages.remove(random_pivot_passage)
            
            #Sets the frontier coordinate to a passage
            self.area[random_pivot_passage.x, random_pivot_passage.y] = Passage(random_pivot_passage.x, random_pivot_passage.y, self.area)
            self.passages.append(self.area[random_pivot_passage.x, random_pivot_passage.y])
            self.pivot_passages.append(self.area[random_pivot_passage.x, random_pivot_passage.y])

            #updates the number of frontier positions left
            self.frontier_passages.update(self.area[random_pivot_passage.x, random_pivot_passage.y].connections)
            
        
        i = 0
        while i < self.loops:
            odd_passages = [passage for passage in self.passages if passage.x % 2 == 1 and passage.y % 2 == 1]
            random_passage = random.choice(odd_passages)
            for passage in random_passage.adjacent_passages:
                link_passage = self.area[self.linked_passage(passage, random_passage)]
                
                if isinstance(link_passage, Wall):
                    self.area[self.linked_passage(passage, random_passage)] = Passage(link_passage.x, link_passage.y, self.area)
                    i += 1
                    break
            
            odd_passages.remove(random_passage)
        
#         self.random_starting_position()
#         self.random_end_position()
        
    def convert_area(self, index):
        """
        Converts the Wall and Passage Objects to 1s and 0s
        """
        
        if isinstance(index, Wall):
            index = 1
            return index
        elif isinstance(index, Passage):
            index = 0
            return index
        else:
            return index
            
    def export_df(self):
        """
        Takes in the Maze and exports it as a Pandas DataFrame
        """
        
        csv = []
        for row in self.area:
            csv.append(list(map(self.convert_area, row)))
        return pd.DataFrame(np.array(csv))
        
    def __str__(self) -> str:
        """
        Returns the maze in a string form
        """
        
        maze_str = ""
        
        for row in self.area:
            for obj in row:
                maze_str += obj.__str__() + " "
            
            maze_str += "\n"
            
        return maze_str