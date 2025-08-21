# agent.py

import random
import numpy as np
import pandas as pd
import lightgbm as lgb

history = {
    "action":     [0, 1, 2, 0], # my action records
    "opponent":   [0, 1, 2], # opponent action records
    "input_2seq" : pd.DataFrame([[0, 0, 1, 1]]), # dataframe for 2seq-predictor
    "input_3seq" : pd.DataFrame([[0, 0, 1, 1, 2, 2]]), # dataframe for 3seq-predictor
    "target": pd.Series([0]),
    "target_2" : pd.Series([0]),
    "strategy_weight" : [1, 1, 1, 1,  # 2seq, 3seq, 2seq_counter, 3seq_counter
                         0.5, 0.5, 0.5, 0.5,  # 2seq_2, 3seq_2, 2seq_counter_2, 3seq_counter_2
                         0.25, 0.25, 0.25, 0.25,  # 2seq_3, 3seq_3, 2seq_counter_3, 3seq_counter_3
                         0.1], # random strategy
    "last_strategy" : "random" # my strategy in last match
}

def check_win(my_action, opponent_action):
    """Checks if the agent's action results in a win."""
    if ((my_action == 2 and opponent_action == 1) or
        (my_action == 1 and opponent_action == 0) or
        (my_action == 0 and opponent_action == 2)):
        return True
    return False

def check_lose(my_action, opponent_action):
    """Checks if the agent's action results in a loss."""
    if ((my_action == 2 and opponent_action == 0) or
        (my_action == 1 and opponent_action == 2) or
        (my_action == 0 and opponent_action == 1)):
        return True
    return False

def my_agent(observation, configuration):
    """The main agent function that makes a decision."""

    global history
    
    change_rate = 1.25 # weight change rate of strategy
    sub_change_rate = 1.1 # weight change rate of related strategy

    history['action'].append(action)
    return int(action)