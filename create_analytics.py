import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yaml


def training_analytics():
    training_data = pd.read_csv("C:/Users/Matta/Documents/Python/COMS_527_Final/results/23-11-16-14-26/training_rewards.csv")

    plt.scatter(np.arange(0, len(training_data["Rewards"])), training_data["Rewards"]
                )
    plt.show()

def validate_checkpoints():
    csv_location = "C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/Checkpoints_500eps_2GPU.csv"
    checkpoint_df = pd.read_csv(csv_location)

    frames = checkpoint_df.filter(like="frame")
    rewards = checkpoint_df.filter(like = "reward")

    numeric_part = rewards.columns.str.extract(r'model-(\d+)state_dict_reward', expand = False).astype(int)

    rewards = rewards[sorted(rewards.columns, key=lambda x: numeric_part[rewards.columns.get_loc(x)])]

    frames = frames[sorted(frames.columns, key=lambda x: numeric_part[frames.columns.get_loc(x)])]

    plt.scatter(rewards.sum().index, rewards.sum().values)
    plt.xlabel("episodes trained (x50)")
    plt.ylabel("reward (sum 10 tests)")
    plt.title("Test rewards (summed)")
    plt.show()

    plt.scatter(frames.sum().index, frames.sum().values)
    plt.xlabel("episodes trained (x50)")
    plt.ylabel("Frames (sum 10 tests)")
    plt.title("Test rewards (summed)")
    plt.show()




    

def validation_analytics():
    pass



if __name__ == "__main__":
    # training_analytics()
    validate_checkpoints()