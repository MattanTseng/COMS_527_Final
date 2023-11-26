import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np 
from typing import List

def training_analytics():
    training_data = pd.read_csv("C:/Users/Matta/Documents/Python/COMS_527_Final/results/23-11-16-14-26/training_rewards.csv")

    plt.scatter(np.arange(0, len(training_data["Rewards"])), training_data["Rewards"]
                )
    plt.show()

def validate_checkpoints(csv_location: str):
    checkpoint_df = pd.read_csv(csv_location)

    frames = checkpoint_df.filter(like="frame")
    rewards = checkpoint_df.filter(like = "reward")

    numeric_part = rewards.columns.str.extract(r'model-(\d+)state_dict_reward', expand = False).astype(int)

    rewards = rewards[sorted(rewards.columns, key=lambda x: numeric_part[rewards.columns.get_loc(x)])]

    frames = frames[sorted(frames.columns, key=lambda x: numeric_part[frames.columns.get_loc(x)])]

    plt.figure()
    plt.scatter(range(len(rewards.columns)), rewards.mean().values)
    plt.xlabel("episodes trained (x50)")
    plt.ylabel("reward (sum 10 tests)")
    plt.title("Test rewards (mean)")

    fig_name = str(os.path.splitext(os.path.basename(csv_location))[0]) + "rewards_wrtEp.png"
    plt.savefig(os.path.join(os.path.dirname(csv_location), fig_name))

    plt.figure()

    plt.scatter(range(len(frames.columns)), frames.mean().values)
    plt.xlabel("episodes trained (x50)")
    plt.ylabel("Frames (sum 10 tests)")
    plt.title("Test frames (mean)")
    fig_name = str(os.path.splitext(os.path.basename(csv_location))[0]) + "frames_wrtEp.png"
    plt.savefig(os.path.join(os.path.dirname(csv_location), fig_name))


def compare_training_checkpoints(csv_1_location: str, csv_2_location: str):
    checkpoint_1_df = pd.read_csv(csv_1_location)
    checkpoint_2_df = pd.read_csv(csv_2_location)

    frames_1 = checkpoint_1_df.filter(like="frame")
    frames_2 = checkpoint_2_df.filter(like="frame")

    numeric_part = frames_1.columns.str.extract(r'model-(\d+)state_dict_frame', expand = False).astype(int)
    frames_1 = frames_1[sorted(frames_1.columns, key=lambda x: numeric_part[frames_1.columns.get_loc(x)])]
    frames_2 = frames_2[sorted(frames_2.columns, key=lambda x: numeric_part[frames_2.columns.get_loc(x)])]

    frames_1_mean = frames_1.mean()
    frames_2_mean = frames_2.mean()

    new_x_labels = range(len(frames_1.columns.tolist()))
    new_x_labels = [str(i) for i in new_x_labels]


    plt.figure()
    ax = frames_1_mean.plot(kind='bar', position = 0, width = 0.4, color = "royalblue", label = "Benchmark")
    frames_2_mean.plot(kind='bar', position = 1, width = 0.4, ax=ax, color = "darkorange",  label="Parallel")

    plt.title("Checkpoint Validation Comparison")
    plt.ylabel("Mean Frame Duration")
    plt.xlabel("Training Checkpoint")
    plt.legend()

    ax.set_xticks(np.arange(len(new_x_labels)), tuple(new_x_labels))
    fig_name = "Final_Results/checkpoint_performance_compare.png"
    print(fig_name)
    plt.savefig(os.path.join(os.path.dirname(csv_1_location[0]), fig_name))


def checkpoint_frame_difference(csv_1_location: str, csv_2_location: str):
    checkpoint_1_df = pd.read_csv(csv_1_location)
    checkpoint_2_df = pd.read_csv(csv_2_location)

    frames_1 = checkpoint_1_df.filter(like="frame")
    frames_2 = checkpoint_2_df.filter(like="frame")

    numeric_part = frames_1.columns.str.extract(r'model-(\d+)state_dict_frame', expand = False).astype(int)
    frames_1 = frames_1[sorted(frames_1.columns, key=lambda x: numeric_part[frames_1.columns.get_loc(x)])]
    frames_2 = frames_2[sorted(frames_2.columns, key=lambda x: numeric_part[frames_2.columns.get_loc(x)])]

    frames_1_mean = frames_1.mean()
    frames_2_mean = frames_2.mean()

    frames_1_mean = frames_1_mean.values
    frames_2_mean = frames_2_mean.values

    frames_differences = np.subtract(frames_2_mean, frames_1_mean)
    color = []

    for difference in frames_differences:
        if difference < 0:
            color = color + ["royalblue"]
        else:
            color = color + ["darkorange"]
    new_x_labels = range(len(frames_differences))
    new_x_labels = [str(i) for i in new_x_labels]

    plt.figure()
    plt.bar(new_x_labels, frames_differences, color = color)
    plt.title("Difference Between Benchmark/Parallel Performance")
    plt.xlabel("Training Checkpoint")
    plt.ylabel("Mean Frame Difference")

    fig_name = "Final_Results/checkpoint_frame_difference.png"
    plt.savefig(os.path.join(os.path.dirname(csv_1_location[0]), fig_name))






    # frames_differences_df = pd.DataFrame(frames_differences, columns=new_x_labels)
    # frames_differences_df['this_color'] = frames_differences_df[]

    # plt.figure()
    # ax = frames_differences_df.plot(kind='bar', position = 0, width = 0.4, color = "royalblue", label = "Frame Difference")

    # plt.title("Checkpoint Validation Comparison")
    # plt.ylabel("Mean Frame Duration")
    # plt.xlabel("Training Checkpoint")
    # plt.legend()

    # ax.set_xticks(np.arange(len(new_x_labels)), tuple(new_x_labels))


def validate_checkpoints_wrt_time(csv_location: str, yaml_location: str):
    with open(yaml_location, 'r') as time_file:
        yaml_data = yaml.safe_load(time_file)

    checkpoint_times = yaml_data["Checkpoint Time"]

    checkpoint_times = np.array(checkpoint_times)

    checkpoint_time_from_start = checkpoint_times - yaml_data["Start Time"]


    checkpoint_df = pd.read_csv(csv_location)

    frames = checkpoint_df.filter(like="frame")
    rewards = checkpoint_df.filter(like = "reward")

    numeric_part = rewards.columns.str.extract(r'model-(\d+)state_dict_reward', expand = False).astype(int)

    rewards = rewards[sorted(rewards.columns, key=lambda x: numeric_part[rewards.columns.get_loc(x)])]

    frames = frames[sorted(frames.columns, key=lambda x: numeric_part[frames.columns.get_loc(x)])]

    plt.figure()
    plt.scatter(checkpoint_time_from_start, rewards.mean().values)
    plt.xlabel("Training Time (s)")
    plt.ylabel("Rewards")
    plt.title("Checkpoint Rewards (mean)")

    fig_name = str(os.path.splitext(os.path.basename(csv_location))[0]) + "rewards_wrtTime.png"
    plt.savefig(os.path.join(os.path.dirname(csv_location), fig_name))

    plt.figure()

    plt.scatter(checkpoint_time_from_start, frames.mean().values)
    plt.xlabel("Training Time (s)")
    plt.ylabel("Frames")
    plt.title("Checkpoint Duration (mean)")
    fig_name = str(os.path.splitext(os.path.basename(csv_location))[0]) + "frames_wrtTime.png"
    plt.savefig(os.path.join(os.path.dirname(csv_location), fig_name))


    return checkpoint_time_from_start

def backprop_time_vs_total_time(yaml_location: str):
    with open(yaml_location, 'r') as time_file:
        yaml_data = yaml.safe_load(time_file)

    total_time = yaml_data["End Time"] - yaml_data["Start Time"]

    backProp_time = yaml_data["Back Prop Time"]
    simulation_time = total_time - backProp_time

    labels = ["Simulation", "Update"]
    slices = [simulation_time, backProp_time]
    plt.figure()
    plt.pie(slices, labels=labels, autopct='%1.1f%%')

    plt.title("Ratio of Simulation Time to Model Update Time")

    fig_name = str(os.path.splitext(os.path.basename(yaml_location))[0]) + "_time_ratio.png"
    plt.savefig(os.path.join(os.path.dirname(yaml_location), fig_name))

def bar_chart_time_compare(time_locations: List[str], labels: List[str]):
    total_times = []
    update_times = []

    time_types = ["simulation_time", "update_time"]
    
    for location in time_locations:
        with open(location, 'r') as time_file:
            yaml_data = yaml.safe_load(time_file)
        this_update_time = yaml_data["Back Prop Time"]
        this_total_time = yaml_data["End Time"] - yaml_data["Start Time"]

        total_times = total_times + [this_total_time]
        update_times = update_times + [this_update_time]


    total_times = np.array(total_times)
    update_times = np.array(update_times)
    simulation_times = total_times - update_times

    plt.figure()

    for label in labels:
        print("Label: ", label)
        for i in range(len(simulation_times)):
            plt.bar(label, simulation_times[i], width=0.4)
            plt.bar(label, update_times[i], bottom= simulation_times[i], width=0.4)

    
    fig_name = str(os.path.splitext(os.path.basename(time_locations[0]))[0]) + "_time_bar_compare.png"
    plt.savefig(os.path.join(os.path.dirname(time_locations[0]), fig_name))

def plot_checkpoint_times(yaml1_loc: str, csv1_loc: str, yaml2_loc: str, csv2_loc: str):
    with open(yaml1_loc, 'r') as time_1_file:
        time_1 = yaml.safe_load(time_1_file)

    with open(yaml2_loc, 'r') as time_2_file:
        time_2 = yaml.safe_load(time_2_file)

    checkpoint_1_df = pd.read_csv(csv1_loc)
    checkpoint_2_df = pd.read_csv(csv2_loc)

    frames_1 = checkpoint_1_df.filter(like="frame")
    frames_2 = checkpoint_2_df.filter(like="frame")

    start_time_1 = time_1["Start Time"]
    start_time_2 = time_2["Start Time"]

    frames_1_times = np.array(time_1["Checkpoint Time"]) - start_time_1
    frames_2_times = np.array(time_2["Checkpoint Time"]) - start_time_2

    plt.figure()
    ax = plt.bar(np.arange(len(frames_2_times)), frames_2_times, width = 0.4, label = "Parallelized Times", color = "darkorange")

    plt.bar(np.arange(len(frames_1_times)) + 0.4, frames_1_times, width = 0.4, label = "Benchmark Times", color = "royalblue")
    plt.title("Checkpoint Creation Time Comparison")
    plt.legend()

    plt.xticks(np.arange(len(frames_1_times)), tuple(np.arange(len(frames_1_times)).astype(str)))


    plt.ylabel("Time (s)")
    plt.xlabel("Training Checkpoint")

    fig_name = "Final_Results/checkpoint_time_compare.png"
    print(fig_name)
    plt.savefig(os.path.join(os.path.dirname(csv1_loc[0]), fig_name))


if __name__ == "__main__":
    benchmark_CSV = "C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/BenchmarkData/checkpoint_comparison_1000eps_1GPU.csv"
    benchmark_YAML = "C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/BenchmarkData/Benchmark_TimeStats.YAML"

    compare_CSV = "C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/2_GPUs/checkpoint_comparison_2GPUs.csv"
    compare_YAML = "C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/2_GPUs/23-11-22-15-21/0time_stats.YAML"
    # training_analytics()
    validate_checkpoints(benchmark_CSV)
    validate_checkpoints_wrt_time(benchmark_CSV, benchmark_YAML)
    backprop_time_vs_total_time(benchmark_YAML)

    validate_checkpoints(compare_CSV)
    validate_checkpoints_wrt_time(compare_CSV, compare_YAML)
    backprop_time_vs_total_time(compare_YAML)

    plot_checkpoint_times(benchmark_YAML, benchmark_CSV, compare_YAML, compare_CSV)
    checkpoint_frame_difference(benchmark_CSV, compare_CSV)
    compare_training_checkpoints(benchmark_CSV, compare_CSV)

    bar_chart_time_compare([benchmark_YAML, compare_YAML], ["1", "2"])
    # validate_checkpoints("C:/Users/Matta/Documents/Python/COMS_527_Final/Final_Results/Checkpoints_500eps_2GPU.csv")

    
