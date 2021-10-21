import argparse
import d4rl
import gym
import tqdm
import numpy as np
import time

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
    print(env.sim.data.qpos)

    oculus_reader = OculusReader()

    # keyboard = KeyboardDevice({})
    # dataset = env.get_dataset()
    # actions = dataset['actions']
    # env.set_target(TARGET_POS)
    # env.reset_to_location(START_POS)
    last_pose = None
    while True:

        # init_qpos = env.init_qpos
        # env.robot.make_controller(env.sim)
        # init_qpos[:7] = env.robot._controller.joint_pos
        # env.robot.reset(env, init_qpos, np.zeros_like(init_qpos))
        # env.render(mode='human')
        # #
        # eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("end_effector")]
        # eef_ori_mat1 = env.sim.data.site_xmat[env.sim.model.site_name2id("end_effector")].reshape([3, 3])
        # print("Init EEF Pos: ", eef_pos)
        # print("Init EEF Orientation: ", eef_ori_mat1)
        # # time.sleep(1)
        # #
        # env = gym.make(args.env_name)
        # env.reset()
        # _, _, done, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
        #
        # init_qpos[:7] = env.robot._controller.commanded_joint_positions
        # env.robot.reset(env, init_qpos, np.zeros_like(init_qpos))
        #
        # env.render(mode='human')
        #
        # eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("end_effector")]
        # eef_ori_mat2 = env.sim.data.site_xmat[env.sim.model.site_name2id("end_effector")].reshape([3, 3])
        #
        # print("")
        # print("Target EEF Pos: ", eef_pos)
        # print("Init EEF Orientation: ", eef_ori_mat2)
        #
        # ori_error = T.get_orientation_error(T.mat2quat(eef_ori_mat2), T.mat2quat(eef_ori_mat1))
        # print("")
        # print(ori_error)
        # print(T.mat2quat(T.euler2mat(ori_error)))
        #
        # time.sleep(1000)

        oculus_pose_data, oculus_button_data = oculus_reader.get_transformations_and_buttons()

        if oculus_button_data and 'rightTrig' in oculus_button_data and oculus_button_data['rightTrig'][0]:
            gripper_action = np.asarray([0])    # close gripper upon button press
        else:
            gripper_action = np.asarray([1])

        joint_action = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32)
        if oculus_button_data and 'rightGrip' in oculus_button_data and oculus_button_data['rightGrip'][0] \
              and oculus_pose_data and 'r' in oculus_pose_data:
            if last_pose is None:
                last_pose = oculus_pose_data['r']
            current_pose = oculus_pose_data['r']

            delta_trans = last_pose[:3, 3] - current_pose[:3, 3]
            # print(current_pose[:3, 3].transpose())

            delta_trans[0], delta_trans[1], delta_trans[2] = \
                        -1 * delta_trans[2], -1 * delta_trans[0], -1 * delta_trans[1]
            # coordinate alignment
            # if env.ROBOTS['robot'].ends_with('OSPControl'):
            #     delta_trans[0], delta_trans[1], delta_trans[2] = \
            #         1 * delta_trans[0], -1 * delta_trans[2], -1 * delta_trans[1]
            # elif env.ROBOTS['robot'].ends_with('IKControl'):
            #     delta_trans[0], delta_trans[1], delta_trans[2] = \
            #         1 * delta_trans[0], -1 * delta_trans[2], -1 * delta_trans[1]
            # else:
            #     raise ValueError("Do not support control for {}".format(env.ROBOTS['robot']))

            joint_action[:3] = 10 * delta_trans
            last_pose = current_pose
        elif oculus_button_data and 'rightGrip' in oculus_button_data and not oculus_button_data['rightGrip'][0]:
            last_pose = None

        action = np.concatenate([joint_action, gripper_action])


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

        # time.sleep(1/100)

