from agent import MineflayerAgent
from dqn import DQNModel
from visualnavigationmodel import VisualNavigationModel

import configuration

from time import sleep
import torch

import maze_task
import maze

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = MineflayerAgent(viewer_port = 3007,name = "yzy1")

model = VisualNavigationModel(7)

mymaze = maze.Maze(width=configuration.MAZE_SIZE, height=configuration.MAZE_SIZE, seed=configuration.MAZE_SEED)

maze_task.build_maze(agent, mymaze)

for r in range(5):
    maze_task.reset_task(agent, mymaze)
    agent.bot.chat("Round " + str(r))
    agentpov = agent.get_image()
    state=torch.tensor(agentpov.flatten(),dtype=torch.float32, device=device)
    last_action = torch.zeros([1,1], dtype=torch.float32, device=device)
    last_reward = torch.zeros([1,1], dtype=torch.float32, device=device)
    for i in range(100):
        # bot actions
        action = model.get_result(state, last_action, last_reward)
        
        agent.apply_action(maze_task.generate_action(action))
        
        # wait for the actions
        sleep(.5)
        agent.stop_movement()

        # calculate rewards
        reward = maze_task.reward(agent, mymaze)
        reward = torch.tensor([[reward]], device=device)

        observation = agent.get_image()
        next_state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device)

        # Store the transition in memory
        model.save_to_memory(state, int(action), next_state, reward, last_action, last_reward)

        # Move to the next state
        state = next_state
        last_action = torch.tensor([[action]], device=device)
        last_reward = reward
        
        if i%10==9:
            msgtxt=str(r)+'-'+str(i)
            print(msgtxt,reward)

        # Perform one step of the optimization
        model.optimize()

sleep(5)

maze_task.clear_maze(agent, mymaze)

# model.save_network('RL_visualnavigation.pt')

del agent