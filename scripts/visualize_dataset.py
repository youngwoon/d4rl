import argparse
import d4rl
import gym
import numpy as np
import os


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    dataset = env.get_dataset()
    rewards = dataset['rewards']
    actions = dataset['actions']

    # NSAMPLES = 1000000
    # import matplotlib.pyplot as plt
    # plt.scatter(qpos[:NSAMPLES, 0], qpos[:NSAMPLES, 1])
    # plt.axis('equal')
    # plt.show()

    ### NORMAL MUJOCO ###
    # if 'infos/qpos' not in dataset:
    #     raise ValueError('Only MuJoCo-based environments can be visualized')
    # qpos = dataset['infos/qpos']
    # qvel = dataset['infos/qvel']
    # env.reset()
    # env.set_state(qpos[0], qvel[0])
    # [env.render() for _ in range(100)]
    # import time; time.sleep(2)
    # for t in range(qpos.shape[0]):
    #     env.set_state(qpos[t], qvel[t])
    #     env.render()

    ### KITCHEN ###
    # split dataset into sequences
    seq_end_idxs = np.where(dataset['terminals'])[0]
    start = 0
    seqs = []
    for end_idx in seq_end_idxs:
        seqs.append(dict(
            states=dataset['observations'][start:end_idx + 1, :30],
            actions=dataset['actions'][start:end_idx + 1],
        ))
        start = end_idx + 1

    # for i, seq in enumerate(seqs):
    #     # if i % 20 > 0: continue
    #     imgs = []
    #     env.reset()
    #     env.set_state(seq['states'][0])
    #     # [env.render() for _ in range(100)]
    #     prev_state = None
    #
    #     # shuffle actions in blocks of N
    #     import random
    #     N = 50
    #     blocks = [seq['actions'][i:i + N] for i in range(0, len(seq['actions']), N)]
    #     random.shuffle(blocks)
    #     seq['actions'][:] = [b for bs in blocks for b in bs]
    #
    #     for state, action in zip(seq['states'], seq['actions']):
    #         if prev_state is not None:
    #             state = 0.2*state + 0.8 * prev_state
    #         # env.step(env.action_space.sample())
    #         env.step(action)
    #         # env.set_state(state)
    #         imgs.append(env.render(mode='rgb_array'))
    #         prev_state = state
    #
    #     # store rendering to disk
    #     save_video(imgs, os.path.join('.', 'kitchen_randSkill50_{}.mp4'.format(i)), fps=15)

    # extract subsequences from specified demos
    SUBSEQS = {0: 60, 60: 75, 507: 75}
    img_seqs = []
    for seq_idx in SUBSEQS:
        seq = seqs[seq_idx]
        imgs = []
        env.reset()
        env.set_state(seq['states'][0])
        prev_state = None
        for state in seq['states'][:SUBSEQS[seq_idx]]:
            if prev_state is not None:
                state = 0.2*state + 0.8 * prev_state
            env.set_state(state)
            imgs.append(env.render(mode='rgb_array'))
            prev_state = state
        img_seqs.append(imgs)

    # fuse all sequences into joint video
    PADDING = 0
    joint_len = np.sum([SUBSEQS[seq_idx] for seq_idx in SUBSEQS]) + 2*PADDING
    joint_frames_1 = img_seqs[0] + [img_seqs[0][-1]] * (joint_len - SUBSEQS[0])
    joint_frames_2 = [img_seqs[1][0]] * (len(img_seqs[0]) + PADDING) + img_seqs[1] + [img_seqs[1][-1]] \
                     * (joint_len - (len(img_seqs[0]) + PADDING) - len(img_seqs[1]))
    joint_frames_3 = [img_seqs[2][0]] * (joint_len - len(img_seqs[2])) + img_seqs[2]

    fused_frames = [np.asarray((jf1/255.+jf2/255.+jf3/255.)/3*255., dtype=np.uint8) for jf1, jf2, jf3 in zip(joint_frames_1, joint_frames_2, joint_frames_3)]
    save_video(fused_frames, os.path.join('.', 'kitchen_demo_fused_noPad.mp4'), fps=25)

