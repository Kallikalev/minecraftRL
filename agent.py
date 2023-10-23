from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from io import BytesIO
from javascript import require, On
from time import sleep

import cv2 as cv
from PIL import Image
import numpy as np

# user-defined configuration
import configuration

mineflayer = require('mineflayer')
pathfinder = require('mineflayer-pathfinder')
mineflayerViewer = require('prismarine-viewer').mineflayer

firefox_options = Options()
# comment the line below to display firefox window
firefox_options.add_argument('-headless')

class MineflayerAgent:
    def __init__(self, viewer_port, name="mineflayer"):
        self.driver = webdriver.Firefox( options = firefox_options)
        self.bot = mineflayer.createBot({
            'host': configuration.SERVER_IP,
            'port': configuration.SERVER_PORT,
            'username': name
        })
        # wait for bot to connect to server
        sleep(1)
        self.viewer = mineflayerViewer(self.bot,{'port':viewer_port,'firstPerson':True})
        # wait for hidden firefox window to load
        sleep(3)
        self.driver.get('http://127.0.0.1:' + str(viewer_port))

    # closes the firefox window when the agent is removed
    # always call this at the end of your program, or invisible firefox processes will stay running
    def __del__(self):
        self.bot.end()
        self.driver.quit()

    def heal(self):
        self.bot.chat('/effect ' + self.bot.username + ' minecraft:instant_health 1 10 true')

    def set_gamemode(self, gamemode):
        self.bot.chat('/gamemode ' + str(gamemode))
    
    def teleport(self, x, y, z):
        self.bot.chat('/tp ' + str(x) + ' ' + str(y) + ' ' + str(z))

    def stop_movement(self):
        self.bot.clearControlStates()

    def apply_action(self, action):
        # action is a vector of length 6, which specifies the following actions:
        # [move, jump, sprint, sneak, turn, attack]

        # movement
        if action[0] >= 1 and action[0] < 5: # if the value is less than 0 don't move, greater than 5 is invalid action
            directions = {1:'back', 2:'left', 3:'right', 4:'forward'}
            choice = directions[int(action[0])] # round down to choose the direction
            self.bot.setControlState(choice, True)
        # jumping
        if action[1] > 0:
            self.bot.setControlState('jump', True)
        # sprinting
        if action[2] > 0:
            self.bot.setControlState('sprint', True)
        # sneaking
        if action[3] > 0:
            self.bot.setControlState('sneak', True)
        # turning
        if action[4] != 0:
            yaw = self.bot.entity.yaw + action[4]
            self.bot.look(yaw, 0)
        # attacking
        if action[5] > 0:
            nearestEntity = self.bot.nearestEntity()
            if nearestEntity is not None:
                self.bot.attack(nearestEntity)

    # takes a screenshot from the firefox window, resizes it according to the configuration
    def get_image(self):
        png = self.driver.get_screenshot_as_png()
        im = Image.open(BytesIO(png)).convert('RGB')
        return cv.resize(np.asarray(im),(configuration.SCREENSHOT_X_RES,configuration.SCREENSHOT_Y_RES))