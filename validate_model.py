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


def find_models(dir_path: str):
    all_files = os.listdir(dir_path)
    pth_files = [file for file in all_files if file.endswith(".pth")]

    return pth_files



def test_models_in_dir(dir_path: str):
    pth_files = find_models(dir_path)

    results = pd.DataFrame()

    for model in pth_files:
        model_name = str(os.path.splitext(model))
        reward_col = model_name + "_reward"
        frame_col = model_name + "_reward"

        rewards, frames = ai_play(os.path.join(dir_path, model))

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



if __name__ == "__main__":
    test_models_in_dir("C:/Users/Matta/Documents/Python/COMS_527_Final/results/23-11-12-12-37")
    
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
        

    
