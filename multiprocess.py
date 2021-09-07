import multiprocessing

import multiprocessing.connection
import gym
import numpy as np
import random


# Multiprocessing setup

class Game(object):
    def __init__(self, game):
        self.env = gym.make(game)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def step(self, action):
        #self.env.render()
        return self.env.step(action)


def runner_process(remote, game):
    game = Game(game) 
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Runner:
    def __init__(self, game):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=runner_process, args=(parent, game))
        self.process.start()