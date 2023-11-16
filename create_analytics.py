import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def training_analytics():
    training_data = pd.read_csv("C:/Users/Matta/Documents/Python/COMS_527_Final/results/23-11-16-14-26/training_rewards.csv")

    plt.scatter(np.arange(0, len(training_data["Rewards"])), training_data["Rewards"])
    plt.show()

def validation_analytics():
    pass



if __name__ == "__main__":
    training_analytics()