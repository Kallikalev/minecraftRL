from agent import MineflayerAgent
import numpy as np
from time import sleep

# gets the agents ready for another round
def reset_task(follower: MineflayerAgent, stander: MineflayerAgent):
    follower.set_gamemode(0)
    stander.set_gamemode(0)

    follower.bot.look(0,0)
    follower.teleport(-2002, 4, -2002)

    randomx = -2028 - np.random.randint(-10,0)
    randomz = -2028 - np.random.randint(-10,0)
    stander.teleport(randomx, 4, randomz)
    
    follower.stop_movement()
    stander.stop_movement()

    sleep(1)
    follower.bot.chat(follower.bot.username+': Ready.')
    stander.bot.chat(stander.bot.username+': Ready.')

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

def reward(follower: MineflayerAgent, stander: MineflayerAgent):
    fPosition = follower.bot.entity.position
    sPosition = stander.bot.entity.position

    x = fPosition.x - sPosition.x
    y = fPosition.y - sPosition.y
    z = fPosition.z - sPosition.z

    # pythagorean theorem
    dist = (x ** 2 + y ** 2 + z ** 2) ** (.5)
    
    angle1 = follower.bot.entity.yaw % (2 * np.pi)
    angle2 = (-np.pi / 2 - np.arctan2(-z, -x)) % (2 * np.pi)

    anglerr = 1 / (1 + (4 / np.pi * (angle1 - angle2)) ** 3) 

    return float(-10 * dist + 50 * anglerr)