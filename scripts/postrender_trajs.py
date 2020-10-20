import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gym
import d4rl

from d4rl.pointmaze.maze_model import MazeEnv
from d4rl.pointmaze.maze_layouts import rand_layout

# from core.rl.envs.maze import MazeEnv


ROLLOUT_PATH = "/Users/karl/Downloads/maze_rollouts/rollout_2.h5"
RES = 512
START_POS = np.array([10., 24.])
TARGET_POS = np.array([18., 8.])


def save_video(file_name, frames, fps=20, video_format='mp4'):
    import skvideo.io
    skvideo.io.vwrite(
        file_name,
        frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


# class RandMaze0S40Env(MazeEnv):
#     START_POS = np.array([10., 24.])
#     TARGET_POS = np.array([18., 8.])
#
#     def _default_hparams(self):
#         default_dict = ParamDict({
#             'name': "maze2d-randMaze0S40-v0",
#         })
#         return super()._default_hparams().overwrite(default_dict)


# load trajectory
with h5py.File(ROLLOUT_PATH, 'r') as F:
    states = F['traj0/states'][()][:, :4]

# plt.scatter(states[:, 0], states[:, 1])
# plt.show()

# render image sequence with environment
kwargs={
        'maze_spec': rand_layout(seed=0, size=40),
        'agent_centric_view': False,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
env = MazeEnv(**kwargs)
env.reset()
env.set_target(TARGET_POS)

imgs = []
for state in states[:950]:
    env.reset_to_location(state[:2])
    env.render(mode='rgb_array')  # these are necessary to make sure new state is rendered on first frame
    env.step(np.zeros_like(env.action_space.sample()))
    img = env.render(mode='rgb_array', width=RES, height=RES)
    imgs.append(img)

save_video('rollout_maze.mp4', imgs, fps=50)

x = 0