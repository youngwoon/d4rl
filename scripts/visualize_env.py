import argparse
import d4rl
import gym


START_POS = [18., 2.]
TARGET_POS = [6., 18.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-randMaze42-ac-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    env.set_target(TARGET_POS)
    env.reset_to_location(START_POS)
    for t in range(10000):
        _, _, done, _ = env.step(env.action_space.sample())
        env.render(mode='human')
        if done:
            env.reset()
            env.set_target(TARGET_POS)
            env.reset_to_location(START_POS)

