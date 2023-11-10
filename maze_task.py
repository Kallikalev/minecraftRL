from agent import MineflayerAgent
from maze import Maze

def reset_task(traveler: MineflayerAgent, maze: Maze):
    startX, startY = (maze.start)
    
    traveler.set_gamemode(0)
    traveler.bot.look(0,0)
    traveler.teleport(startX, 4, startY)

def build_maze(traveler: MineflayerAgent, maze: Maze):
    
    df = maze.export_df().to_numpy()
    sizeX, sizeY = (maze.width, maze.height)
    
    traveler.bot.chat(f"/fill 0 3 0 {sizeX - 1} 3 {sizeY - 1} stone")
    for i, row in enumerate(df):
        for j, block in enumerate(row):
            if block == 1:
                traveler.bot.chat(f"/fill {i} 4 {j} {i} 6 {j} bedrock") 
            if block == 2:
                traveler.bot.chat(f"/fill {i} 3 {j} {i} 5 {j} wool 13") 
            if block == 3:
                traveler.bot.chat(f"/fill {i} 3 {j} {i} 3 {j} wool 14") 

def clear_maze(traveler: MineflayerAgent, maze: Maze):
    sizeX, sizeY = (maze.width, maze.height)
    
    traveler.bot.chat(f"/fill 0 3 0 {sizeX - 1} 3 {sizeY - 1} grass")
    traveler.bot.chat(f"/fill 0 4 0 {sizeX - 1} 6 {sizeY - 1} air")