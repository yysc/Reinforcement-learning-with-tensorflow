import numpy as np 
import pandas as pd 
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions,
    )
    return table

def choose_action(state, q_table):
    state_actions=q_table.iloc[state,:]