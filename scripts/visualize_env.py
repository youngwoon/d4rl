import argparse
import d4rl
import gym
import tqdm
import cv2


START_POS = [27., 4.]
TARGET_POS = [18., 8.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-randMaze0S40-ac-v0')  #'kitchen-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    # dataset = env.get_dataset()
    # actions = dataset['actions']
    # env.set_target(TARGET_POS)
    # env.reset_to_location(START_POS)
    for t in tqdm.tqdm(range(10000)):
        # _, _, done, _ = env.step(actions[t])
        _, _, done, _ = env.step(env.action_space.sample())
        #env.render(mode='human')
        img = env.render(mode='rgb_array')
        if t % 20 == 0:
            cv2.imwrite("test_img_{}.png".format(t), img[:, :, ::-1])
        if done:
            env.reset()
            # env.set_target(TARGET_POS)
            # env.reset_to_location(START_POS)

