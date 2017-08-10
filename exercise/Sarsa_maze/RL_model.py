import numpy as np
import pandas as pd


class SarsaLambaTable(object):

    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space

    def choose_action(state, q_table):
        self.check_state_exists(state)
        if np.random.uniform()>self.e
