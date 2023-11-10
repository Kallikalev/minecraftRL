from agent import MineflayerAgent
from maze import Maze
import numpy as np

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
                traveler.bot.chat(f"/fill {i} 3 {j} {i} 5 {j} gold_block") 
            if block == 3:
                traveler.bot.chat(f"/fill {i} 3 {j} {i} 3 {j} diamond_block") 

def clear_maze(traveler: MineflayerAgent, maze: Maze):
    sizeX, sizeY = (maze.width, maze.height)
    
    traveler.bot.chat(f"/fill 0 3 0 {sizeX - 1} 3 {sizeY - 1} grass")
    traveler.bot.chat(f"/fill 0 4 0 {sizeX - 1} 6 {sizeY - 1} air")

# translates the neural network output vector to an agent action vector
# neural network output is a single integer. 0 is nothing, 1-4 is moving, 5 is turning clockwise, 6 is counterclockwise
def generate_action(network_output):
    action = [0,0,0,0,0,0]
    if network_output < 5:
        action[0] = int(network_output)
    elif network_output == 5: 
        action[4] = -np.pi/8
    else:
        action[4] = np.pi/8
    return action

def reward(traveler: MineflayerAgent, maze: Maze):
    botPosition = traveler.bot.entity.position
    goalX = maze.end[0]
    goalY = maze.end[1]
    goalZ = 4

    x = botPosition.x - goalX
    y = botPosition.y - goalY
    z = botPosition.z - goalZ

    # pythagorean theorem
    dist = (x ** 2 + y ** 2 + z ** 2) ** (.5)
    

    return (10-dist)/10