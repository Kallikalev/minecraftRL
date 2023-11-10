from agent import MineflayerAgent
from dqn import DQNModel
from visualnavigationmodel import VisualNavigationModel

from time import sleep
import torch

import followtask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent1 = MineflayerAgent(viewer_port = 3007,name = "yzy1")
agent2 = MineflayerAgent(viewer_port = 3008,name = "yzy2")

model = VisualNavigationModel(7,'RL_visualnavigation.pt')


for r in range(100):
    followtask.reset_task(agent1, agent2)
    agent1.bot.chat("Round " + str(r))
    agent1pov = agent1.get_image()
    state=torch.tensor(agent1pov.flatten(),dtype=torch.float32, device=device)
    last_action = torch.zeros([1,1], dtype=torch.float32, device=device)
    last_reward = torch.zeros([1,1], dtype=torch.float32, device=device)
    for i in range(20):
        # bot actions
        action = model.get_result(state, last_action, last_reward)
        
        agent1.apply_action(followtask.generate_action(action))
        
        # wait for the actions
        sleep(.5)
        agent1.stop_movement()

        # calculate rewards
        reward = followtask.reward(agent1, agent2)
        reward = torch.tensor([[reward]], device=device)

        observation = agent1.get_image()
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

model.save_network('RL_visualnavigation.pt')

del agent1
del agent2