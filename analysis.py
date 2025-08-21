# analysis.py

import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns


history = {
    "action": [0, 1, 2, 0],
    "opponent": [0, 1, 2],
    "input_2seq": pd.DataFrame([[0, 0, 1, 1]]),
    "input_3seq": pd.DataFrame([[0, 0, 1, 1, 2, 2]]),
    "target": pd.Series([0]),
    "target_2": pd.Series([0]),
    "strategy_weight": [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.1],
    "last_strategy": "random"
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
    """The main agent function."""
    global history

    change_rate = 1.25
    sub_change_rate = 1.1
    confirm_thr = 0.50
    draw_penalty = 1.01
    safe_thr = 0.15
    num_records = 50
    params = {'objective': 'multiclass', 'num_class': 3, 'max_depth': 6, 'num_iterations': 10, 'learning_rate': 0.1,
              'verbosity': -1}

    opponent_action = observation.lastOpponentAction if observation.step > 0 else 0
    history['opponent'].append(opponent_action)

    if observation.step > 0:
        last_my_action = history["action"][-1]
        last_opponent_action = history["opponent"][-1]

        if history['last_strategy'] == "2seq":
            if check_win(last_my_action, last_opponent_action):
                history["strategy_weight"][0] *= change_rate
            elif check_lose(last_my_action, last_opponent_action):
                history["strategy_weight"][0] /= change_rate
            else:
                history["strategy_weight"][0] *= draw_penalty
        # (All other strategy weight update logic would be here)

    if observation.step >= 4:
        new_2seq_row = pd.DataFrame(
            [[history['action'][-3], history['opponent'][-3], history['action'][-2], history['opponent'][-2]]])
        new_3seq_row = pd.DataFrame([[history['action'][-4], history['opponent'][-4], history['action'][-3],
                                      history['opponent'][-3], history['action'][-2], history['opponent'][-2]]])
        history['input_2seq'] = pd.concat([history['input_2seq'], new_2seq_row], ignore_index=True)
        history['input_3seq'] = pd.concat([history['input_3seq'], new_3seq_row], ignore_index=True)
        history['target'] = pd.concat([history['target'], pd.Series([history['opponent'][-1]])], ignore_index=True)
        history['target_2'] = pd.concat([history['target_2'], pd.Series([history['action'][-1]])], ignore_index=True)

    history['input_2seq'] = history['input_2seq'].tail(num_records)
    history['input_3seq'] = history['input_3seq'].tail(num_records)
    history['target'] = history['target'].tail(num_records)
    history['target_2'] = history['target_2'].tail(num_records)

    action = random.randint(0, 2)
    history["last_strategy"] = "random"

    if observation.step >= num_records:
        strategy_names = ["2seq", "3seq", "2seq_counter", "3seq_counter", "2seq_2", "3seq_2", "2seq_counter_2",
                          "3seq_counter_2", "2seq_3", "3seq_3", "2seq_counter_3", "3seq_counter_3", "random"]
        strategy = random.choices(strategy_names, weights=history["strategy_weight"])[0]
        history["last_strategy"] = strategy

        # (The full prediction logic would be here, for simplicity we'll keep the random action as a fallback)
        if "2seq" in strategy and "counter" not in strategy and len(history['input_2seq']) > 10:
            # This is a simplified version of your prediction logic
            pass  # In a real run, your full LGBM prediction code would be here

    history['action'].append(action)
    return int(action)



def random_agent(observation, configuration):
    """A simple agent that plays randomly."""
    return random.randint(0, 2)


Observation = namedtuple('Observation', ['step', 'lastOpponentAction'])
Configuration = namedtuple('Configuration', ['episodeSteps'])


def run_game_and_collect_data(agent1, agent2, num_episodes):
    """Simulates a game and logs detailed data from each turn for analysis."""
    global history
    # Reset history for a fresh run
    history = {
        "action": [0, 1, 2, 0], "opponent": [0, 1, 2],
        "input_2seq": pd.DataFrame([[0, 0, 1, 1]]), "input_3seq": pd.DataFrame([[0, 0, 1, 1, 2, 2]]),
        "target": pd.Series([0]), "target_2": pd.Series([0]),
        "strategy_weight": [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.1],
        "last_strategy": "random"
    }

    log = []
    my_score, opponent_score = 0, 0
    last_action1, last_action2 = 0, 0
    strategy_names = ["2seq", "3seq", "2seq_counter", "3seq_counter", "2seq_2", "3seq_2", "2seq_counter_2",
                      "3seq_counter_2", "2seq_3", "3seq_3", "2seq_counter_3", "3seq_counter_3", "random"]

    for i in range(num_episodes):
        obs1 = Observation(step=i, lastOpponentAction=last_action2)
        action1 = agent1(obs1, None)
        obs2 = Observation(step=i, lastOpponentAction=last_action1)
        action2 = agent2(obs2, None)

        outcome = "draw"
        if check_win(action1, action2):
            my_score += 1
            outcome = "win"
        elif check_lose(action1, action2):
            opponent_score += 1
            outcome = "loss"

        log_entry = {'step': i, 'my_action': action1, 'opponent_action': action2, 'outcome': outcome,
                     'my_cumulative_score': my_score, 'opponent_cumulative_score': opponent_score,
                     'chosen_strategy': history['last_strategy']}
        for idx, name in enumerate(strategy_names):
            log_entry[f'weight_{name}'] = history['strategy_weight'][idx]
        log.append(log_entry)

        last_action1, last_action2 = action1, action2

    return pd.DataFrame(log)


def analyze_results(results_df):
    """Takes the simulation log and generates analytical graphs."""
    final_my_score = results_df['my_cumulative_score'].iloc[-1]
    final_opponent_score = results_df['opponent_cumulative_score'].iloc[-1]
    draws = len(results_df) - final_my_score - final_opponent_score
    print("\n--- Final Score ---")
    print(f"My Agent: {final_my_score} | Opponent: {final_opponent_score} | Draws: {draws}\n")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Analysis of Agent Performance vs. Random Agent', fontsize=20)

    # Plot 1: Performance Over Time
    axes[0, 0].plot(results_df['step'], results_df['my_cumulative_score'], label='My Agent Score', color='royalblue')
    axes[0, 0].plot(results_df['step'], results_df['opponent_cumulative__score'], label='Opponent Score',
                    color='tomato')
    axes[0, 0].set_title('Cumulative Score Over Time', fontsize=14)
    axes[0, 0].set_xlabel('Game Step')
    axes[0, 0].set_ylabel('Total Score')
    axes[0, 0].legend()

    # Plot 2: Strategy Usage Frequency
    strategy_counts = results_df[results_df['step'] >= 50]['chosen_strategy'].value_counts()
    sns.barplot(ax=axes[0, 1], x=strategy_counts.values, y=strategy_counts.index, palette='viridis', orient='h')
    axes[0, 1].set_title('Strategy Usage Frequency (After Step 50)', fontsize=14)
    axes[0, 1].set_xlabel('Number of Times Chosen')
    axes[0, 1].set_ylabel('Strategy')

    # Plot 3: Evolution of Top Strategy Weights
    strategy_names = [col.replace('weight_', '') for col in results_df.columns if 'weight_' in col]
    top_strategies = strategy_counts.nlargest(5).index
    for name in top_strategies:
        axes[1, 0].plot(results_df['step'], results_df[f'weight_{name}'], label=name)
    axes[1, 0].set_title('Evolution of Top 5 Strategy Weights', fontsize=14)
    axes[1, 0].set_xlabel('Game Step')
    axes[1, 0].set_ylabel('Strategy Weight (Log Scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()

    # Plot 4: Action Distribution
    action_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    action_data = results_df[['my_action', 'opponent_action']].copy()
    action_data['my_action'] = action_data['my_action'].map(action_map)
    action_data['opponent_action'] = action_data['opponent_action'].map(action_map)
    plot_data = pd.melt(action_data, var_name='Player', value_name='Action')
    sns.countplot(ax=axes[1, 1], data=plot_data, x='Action', hue='Player',
                  palette={'My Agent': 'skyblue', 'Opponent': 'salmon'})
    axes[1, 1].set_title('Distribution of Actions Chosen', fontsize=14)
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('Total Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



if __name__ == "__main__":
    NUM_EPISODES = 1000
    print(f"Running simulation for {NUM_EPISODES} episodes...")

    results_df = run_game_and_collect_data(my_agent, random_agent, num_episodes=NUM_EPISODES)

    analyze_results(results_df)