from PIL import Image
from collections import deque, namedtuple
import datetime
from itertools import count
import os
import shutil
import gymnasium as gym
from torchvision.utils import torch
import torch.nn as nn
import random
import envs
import numpy as np
import time

from model import DQN
from train import MemoryReplay



from torch.nn.parallel import DistributedDataParallel as DDP

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# parallelization code is coming from https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
# And https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
# https://towardsdatascience.com/distribute-your-pytorch-model-in-less-than-20-lines-of-code-61a786e6e7b0

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)

class Parallel_Trainer:
    def __init__(
        self,
        gpu_id: int,
        env: envs.Wrapper,
        policy_net: DQN,
        target_net: DQN,
        n_episodes=5,
        lr=1e-4,
        batch_size= 32,
        replay_size=10_000,  # experience replay's buffer size
        learning_start= 10_000,  # number of frames before learning starts
        target_update_freq=1_000,  # number of frames between every target network update
        optimize_freq=1,
        gamma=0.99,  # reward decay factor
        # explore/exploit eps-greedy policy
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=10_000,
    ):
        self.env = env
        # There's a few times that I send specific tensors to the gpu. Need to define this string
        # to make that transition easy
        self.gpu_id = gpu_id
        self.device = "cuda:" + str(gpu_id)

        print("Parallel Trainer constructor: ", self.device)
        self.policy_net = policy_net
        self.target_net = target_net



        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory_replay = MemoryReplay(replay_size)


        self.n_steps = 0

        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        self.learning_start = learning_start
        self.target_update_freq = target_update_freq
        self.optimize_freq = optimize_freq

        self.gamma = gamma

        self._get_eps = lambda n_steps: eps_end + (eps_start - eps_end) * np.exp(
            -1.0 * n_steps / eps_decay
        )

        # Initialize folder to save training results
        folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + "gpu_" + str(self.gpu_id)
        folder_path = os.path.join("results", folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        self.folder_path = folder_path

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select the next action given the current state following the eps-greedy policy"""
        eps = self._get_eps(self.n_steps)

        if random.random() > eps:
            # exploit
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).max(dim=1)[1][0]
        else:
            # explore
            return torch.tensor(self.env.action_space.sample(), device=self.device)

    def _optimize(self):
        transitions = self.memory_replay.sample(self.batch_size)

        # Convert batch-array of Transitions to a Transition of batch-arrays
        batch = Transition(*zip(*transitions))






        state_batch = torch.stack(tuple(batch.state[0].to(self.device)))
        # need to reshape now
        state_batch = state_batch.reshape(1, 4, 64, 128)
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.stack(batch.reward)
        terminated_batch = torch.tensor(
            batch.terminated, device=self.device, dtype=torch.float
        )

        # Compute batch "Q(s, a)"
        # The model returns "Q(s)", then we select the columns of actions taken.

        Q_values = (
            self.policy_net(state_batch)
            .gather(1, action_batch.unsqueeze(-1))
            .squeeze(-1)
        )

        # Compute batch "max_{a'} Q(s', a')"
        with torch.no_grad():
            next_Q_values = self.target_net(next_state_batch).max(1)[0]
        expected_Q_values = (
            1.0 - terminated_batch
        ) * next_Q_values * self.gamma + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, expected_Q_values)

        # print(f"Loss: {loss.item()}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode_i in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device)

            total_reward = 0.0

            for t in count():
                self.n_steps += 1

                action = self._select_action(state)

                next_state, reward, terminated, *_ = self.env.step(
                    envs.Action(action.item())
                )
                next_state = torch.tensor(next_state)

                total_reward += float(reward)

                self.memory_replay.push(
                    state,
                    action,
                    next_state,
                    torch.tensor(reward, device=self.device),
                    terminated,
                )

                # Synchronize the target network with the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.target_update_freq == 0
                ):
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Optimize the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.optimize_freq == 0
                ):
                    self._optimize()

                if terminated:
                    print(
                        f"HARDWARE: {self.device} \t episode: {episode_i}\t steps: {t+1}\t reward: {total_reward}"
                    )
                    break
                else:
                    state = next_state
            # only save checkpoint for gpu 0
            if episode_i % 50 == 0 and self.device == "cuda:0":
                self.save_obs_result(episode_i, self.env.frames)
                self.save_model_weights(episode_i)

        self.env.close()

    def save_obs_result(self, episode_i: int, obs_arr: list[np.ndarray]):
        frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
        file_path = os.path.join(self.folder_path, f"episode-{episode_i}.gif")

        frames[0].save(
            file_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def save_model_weights(self, episode_i: int):
        file_path = os.path.join(self.folder_path, f"model-{episode_i}.pth")
        torch.save(self.policy_net, file_path)



def run_single_training(rank, num_gpus):
    print("Running run_single_trainins: ", rank)
    # every training environment needs a gym 
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
    env = envs.Wrapper(env, k=4)

    torch.distributed.init_process_group(backend="gloo", init_method = 'tcp://localhost:12355', rank=rank, world_size=num_gpus)
    torch.cuda.set_device(rank)
    # make sure that there's no data race
    # torch.distributed.barrier()

    # now use attributes of the gym to define the action space
    # Define the DQN networks
    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n

    # these are the models that will be used
    policy_net = DQN(in_channels, out_channels)
    target_net = DQN(in_channels, out_channels)

    # # put the models on the right GPU 
    policy_net = policy_net.cuda(rank)
    target_net = target_net.cuda(rank)

    # setup the models with pytorch distributed data parallel
    policy_net = DDP(policy_net, device_ids=[rank])
    target_net = DDP(target_net, device_ids=[rank])

    trainer = Parallel_Trainer(rank, env, policy_net, target_net)

    trainer.train()

    print("This is run_single_training function")


    


if __name__ == "__main__":
    



    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'

    num_gpus = torch.cuda.device_count()
    print("There are: ", num_gpus, " gpus to be used")
    print("cuda available: ", torch.cuda.is_available())

    if num_gpus == 0:
        print("There are no gpus :(")
        quit()

    print("Starting training")
    tic = time.time()
    # the pytorch multiprocessing spawn takes the function and spins up multiple copies of it on different pieces of hardware
    torch.multiprocessing.spawn(run_single_training, args=(num_gpus, ), nprocs=num_gpus, join = True)

    print("Done training")
    print("Elapsed Training time: ", time.time() - tic)