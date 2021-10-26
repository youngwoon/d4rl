import argparse
import d4rl
import gym
import tqdm
import numpy as np
import time
import glfw

import robosuite.utils.transform_utils as T


from core.data.teleop.devices import KeyboardDevice
from oculus_reader.reader import OculusReader


START_POS = [27., 4.]
TARGET_POS = [18., 8.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-2-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()

    base_act = np.array([-np.pi/2, np.pi/2, 3*np.pi/8, -5*np.pi/8, 0, np.pi/2, np.pi/2, 4.79267505e-02, 3.71350919e-02])

    offset = np.zeros_like(base_act)
    offset_idx = 0
    while True:
        if env.sim_robot.renderer._onscreen_renderer is not None:
            def key_callback(window, key, scancode, action, mods):
                global offset, offset_idx, env
                if action != glfw.RELEASE and action != glfw.PRESS:
                    return
                if key == glfw.KEY_W:
                    offset[offset_idx] += 0.02
                elif key == glfw.KEY_S:
                    offset[offset_idx] -= 0.02
                elif action == glfw.RELEASE and key == glfw.KEY_P:
                    print(env.sim.model.key_qpos[0])
                elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6):
                    offset_idx = key - glfw.KEY_0

            glfw.set_key_callback(env.sim_robot.renderer._onscreen_renderer.window, key_callback)

        _, _, done, _ = env.step(base_act + offset)
        # _, _, done, _ = env.step(np.zeros_like(env.action_space.sample()))
        env.render(mode='human')
        if done:
            env.reset()
            offset = np.zeros_like(offset)
