import argparse
import d4rl
import gym
import tqdm
import numpy as np
import time


from core.data.teleop.devices import KeyboardDevice


START_POS = [27., 4.]
TARGET_POS = [18., 8.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-2-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()

    keyboard = KeyboardDevice({})
    # dataset = env.get_dataset()
    # actions = dataset['actions']
    # env.set_target(TARGET_POS)
    # env.reset_to_location(START_POS)
    for t in tqdm.tqdm(range(10000)):
        action = np.asarray([1, 1, 0, 0, 0, 0, 0])

        # _, d_pose, gripper = keyboard.get_state()
        # d_pose = 100 * d_pose
        # print(d_pose)
        # action = np.concatenate((d_pose, np.asarray(gripper)[None]))

        _, _, done, _ = env.step(action) #env.action_space.sample())
        env.render(mode='human')
        if done:
            env.reset()
            # env.set_target(TARGET_POS)
            # env.reset_to_location(START_POS)

        # time.sleep(1/30)

