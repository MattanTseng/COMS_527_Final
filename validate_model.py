import gymnasium as gym
import pygame
import torch
import envs
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import datetime
import torch.distributed as dist
from model import DQN
import os
import torch.multiprocessing as mp

from MultiEnv import MultiEnv


from model import DQN


def human_play():
    env = gym.make("Env-v0", render_mode="human")
    obs, _ = env.reset()

    total_reward = 0.0
    n_frames = 0
    while True:
        n_frames += 1
        userInput = pygame.key.get_pressed()
        action = envs.Action.STAND
        if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
            action = envs.Action.JUMP
        elif userInput[pygame.K_DOWN]:
            action = envs.Action.DUCK

        obs, reward, terminated, _, _ = env.step(action)

        total_reward += float(reward)
        if terminated:
            break

    print(f"Total reward: {total_reward}, number of frames: {n_frames}")

    env.close()

    # Show image of the last frame
    plt.imshow(obs)
    plt.show()


def play_with_model(
    env: envs.Wrapper,
    policy_net: DQN,
    device: torch.device,
    seed: int | None = None,
) -> float:
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    state = torch.tensor(state, device=device)

    total_reward = 0.0
    while True:
        action = policy_net(state.unsqueeze(0)).max(dim=1)[1][0]

        state, reward, terminated, _, _ = env.step(action)
        state = torch.tensor(state, device=device)

        total_reward += float(reward)
        if terminated:
            break

    return total_reward



# given the path to a model, play the dinosaur game in headless mode. 
def ai_play(model_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore


    env = gym.make("Env-v0", render_mode="headless")
    env = envs.Wrapper(env)

    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n

    # these are the models that will be used
    policy_net = DQN(in_channels, out_channels)
    policy_net.load_state_dict(torch.load(model_path))

    policy_net = policy_net.to(device)

    
    policy_net.eval()

    total_reward = play_with_model(env, policy_net, device)

    print(f"Total reward: {total_reward}, number of frames: {len(env.frames)}")


    env.close()

    return total_reward, len(env.frames)


# function to pull out all state_dict models in a certain directory
def find_models(dir_path: str):
    all_files = os.listdir(dir_path)
    pth_files = [file for file in all_files if file.endswith(".pth")]


    state_dict_pth_files = [s for s in pth_files if "state_dict" in s]


    return state_dict_pth_files



# this function tests the intermediate checkpoints to visualize training progress
def test_models_in_dir(dir_path: str):
    pth_files = find_models(dir_path)

    results = pd.DataFrame()

    for model in pth_files:
        rewards = []
        frames = []
        model_name = str(os.path.splitext(model)[0])
        reward_col = model_name + "_reward"
        frame_col = model_name + "_frame"
        for i in range(0, 10):
            reward, frame = ai_play(os.path.join(dir_path, model))
            rewards = rewards + [reward]
            frames = frames + [frame]

        results[reward_col] = rewards
        results[frame_col] = frames


    results.to_csv("checkpoint_comparison.csv", index=False)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("type", choices=["human", "ai"])
#     parser.add_argument("-m", "--model_path")

#     args = parser.parse_args()
#     if args.type == "human":
#         human_play()
#     else:
#         ai_play(args.model_path)

# https://pytorch.org/docs/stable/notes/multiprocessing.html
# https://discuss.pytorch.org/t/how-do-i-run-inference-in-parallel/126757
# to validate a model, spin up multiple instances at the same time
def run_single_validation(rank, model_path: str, results_q: mp.Queue, instances_per_gpu: int = 1):
    rewards = []
    frames = []
    for _ in range(instances_per_gpu):
        this_reward, this_frame = ai_play(model_path)
        rewards = rewards + [this_reward]
        frames = frames + [this_frame]

    # put the results from the run into the shared memory
    this_result = []
    for i in range(instances_per_gpu):
        this_result = this_result + [(rewards[i], frames[i])]
    
    results_q.put(tuple(this_result))


# https://discuss.pytorch.org/t/how-do-i-run-inference-in-parallel/126757
def validate_parallel(model_location: str, num_gpus: int = 1, instances_per_gpu: int = 1):
    results = mp.Queue()
    
    world_size = num_gpus


    mp.spawn(run_single_validation, 
            args=(model_location,results, instances_per_gpu),
            nprocs=world_size,
            join=True)
    
    # get the results from the shared memory
    cumulative_results = []
    for _ in range(num_gpus):
        cumulative_results.extend(results.get())

    df_to_save = pd.DataFrame(cumulative_results, columns=["Rewards", "Frames"])
    csv_name = "single_model_trials.csv"
    df_to_save.to_csv(csv_name, index=False)

    return cumulative_results
    

if __name__ == "__main__":
    folder_loc = "C:/Users/Matta/Documents/Python/COMS_527_Final/results/23-11-16-15-43/"
    num_gpus = torch.cuda.device_count()
    print("Num gpus: ", num_gpus)

    # run a trial with all models in this folder. 
    test_models_in_dir(folder_loc)

    # test the last trained model
    model_loc = os.path.join(folder_loc, "model-50state_dict.pth")
    print("Spinning up instances in parallel for validation")
    validation_results = validate_parallel(model_loc, num_gpus, 5)

    print("Validation results: ", validation_results)
    # model = sys.argv[1]
    # n_trials = int(sys.argv[2])
    # rewards = []
    # frames = []

    # print(f"Validating for {n_trials} number of trials")

    # for n in range(0, n_trials):
    #     print(f"Trial: {n}")
    #     this_reward, this_frame = ai_play(model)
    #     rewards = rewards + [this_reward]
    #     frames = frames + [this_frame]

    #     # Create a DataFrame
    # df = pd.DataFrame({'Rewards': rewards, 'Frames': frames})

    # output_name = datetime.datetime.now().strftime("%H-%m-%d-%m-%y") + str(".csv")
    # # Save the DataFrame to a CSV file
    # df.to_csv(output_name, index=False)
        

    
