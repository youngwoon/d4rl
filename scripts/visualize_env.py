import argparse
import d4rl
import gym
import tqdm
import numpy as np


START_POS = [27., 4.]
TARGET_POS = [18., 8.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-2-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    # dataset = env.get_dataset()
    # actions = dataset['actions']
    # env.set_target(TARGET_POS)
    # env.reset_to_location(START_POS)
    for t in tqdm.tqdm(range(10000)):
        action = np.asarray([1, 1, 0, 0, 0, 0, 0])
        _, _, done, _ = env.step(action) #env.action_space.sample())
        env.render(mode='human')
        if done:
            env.reset()
            # env.set_target(TARGET_POS)
            # env.reset_to_location(START_POS)

