import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d

# SIZE = 40
#
# RENDER_SCALE = 10
#
# maze_layout = np.zeros((SIZE, SIZE))

### Rand Fill Algorithm

# def fill_block(x, y, maze_layout):
#     if maze_layout[max(x-1, 0), y] or maze_layout[min(x+1, SIZE-1), y] or \
#         maze_layout[x, max(y-1, 0)] or maze_layout[x, min(y+1, SIZE-1)]:
#         return np.random.rand() < 0.5
#     return np.random.rand() < 0.1
#
# for x in range(SIZE):
#     for y in range(SIZE):
#         maze_layout[x, y] = 1 if fill_block(x, y, maze_layout) else 0


# ### Rand Wall-Place Algorithm
# MAX_LEN_FRAC = 0.2      # fraction of total maze size for max wall length
# COVERAGE_FRAC = 0.3     # until what fraction of maze covered we keep adding walls
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max(MAX_LEN_FRAC * SIZE * np.random.rand(), 3))
#     sample_pos1 = int(np.random.rand() * (SIZE - 3 - 1))
#     sample_pos2 = int(np.random.rand() * (SIZE - 1))
#     if sample_vert_hor == 0:
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#     else:
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


# ### Rand Wall-Place Algorithm w/ doors
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.4     # until what fraction of maze covered we keep adding walls
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max((MAX_LEN_FRAC-MIN_LEN_FRAC) * SIZE * np.random.rand() + MIN_LEN_FRAC*SIZE, 3))
#     sample_door_offset = np.random.choice(np.arange(1, sample_len - 1))
#     sample_pos1 = int(np.random.rand() * (SIZE - sample_len - 1))
#     sample_pos2 = int(np.random.rand() * (SIZE - 2) + 1)
#
#     if sample_vert_hor == 0:
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#         maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
#     else:
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#         maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


# ### Rand Wall-Place Algorithm w/ doors and diversified sampling
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.25     # until what fraction of maze covered we keep adding walls
#
#
# def compute_sampling_probs(maze_layout, axis, filter=None):
#     if filter is None:
#         filter = [1/5, 1/5, 1/5, 1/5, 1/5]
#     coverage = np.mean(maze_layout, axis=axis)
#     probs = np.convolve(coverage, np.array(filter), 'valid')
#     return np.exp(1 - probs) / np.sum(np.exp(1 - probs))
#
#
# def place_wall(maze_layout):
#     sample_vert_hor = 0 if np.random.rand() < 0.5 else 1
#     sample_len = int(max((MAX_LEN_FRAC-MIN_LEN_FRAC) * SIZE * np.random.rand() + MIN_LEN_FRAC*SIZE, 3))
#     sample_door_offset = np.random.choice(np.arange(1, sample_len - 1))
#     sample_pos1 = int(np.random.rand() * (SIZE - sample_len - 1))
#
#     # sample_pos2 = int(np.random.rand() * (SIZE - 2) + 1)
#
#     if sample_vert_hor == 0:
#         sample_pos1 = np.random.choice(np.arange(0, SIZE - sample_len+1), p=compute_sampling_probs(
#             maze_layout, axis=1, filter=np.ones(sample_len)/sample_len))
#         sample_pos2 = np.random.choice(np.arange(2, SIZE-2),
#                                        p=compute_sampling_probs(maze_layout[sample_pos1 : sample_pos1 + sample_len, :], axis=0))
#         maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
#         maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
#         maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
#     else:
#         sample_pos1 = np.random.choice(np.arange(0, SIZE - sample_len + 1), p=compute_sampling_probs(
#             maze_layout, axis=0, filter=np.ones(sample_len) / sample_len))
#         sample_pos2 = np.random.choice(np.arange(2, SIZE - 2),
#                                        p=compute_sampling_probs(maze_layout[:, sample_pos1: sample_pos1 + sample_len], axis=1))
#         maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
#         maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
#         maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
#         maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
#     return maze_layout
#
# while np.mean(maze_layout) < COVERAGE_FRAC:
#     maze_layout = place_wall(maze_layout)


### Rand Wall-Place Algorithm w/ doors and diversified sampling in 2D
# MAX_LEN_FRAC = 0.5      # fraction of total maze size for max wall length
# MIN_LEN_FRAC = 0.3
# COVERAGE_FRAC = 0.25     # until what fraction of maze covered we keep adding walls
# TEMP = 20

def compute_sampling_probs(maze_layout, filter, temp):
    probs = convolve2d(maze_layout, filter, 'valid')
    return np.exp(-temp*probs) / np.sum(np.exp(-temp*probs))


def sample_2d(probs, rng):
    flat_probs = probs.flatten()
    sample = rng.choice(np.arange(flat_probs.shape[0]), p=flat_probs)
    sampled_2d = np.zeros_like(flat_probs)
    sampled_2d[sample] = 1
    idxs = np.where(sampled_2d.reshape(probs.shape))
    return idxs[0][0], idxs[1][0]


def place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp):
    size = maze_layout.shape[0]
    sample_vert_hor = 0 if rng.random() < 0.5 else 1
    sample_len = int(max((max_len_frac-min_len_frac) * size * rng.random() + min_len_frac*size, 3))
    sample_door_offset = rng.choice(np.arange(1, sample_len - 1))

    if sample_vert_hor == 0:
        filter = np.ones((sample_len, 5)) / (5*sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[0]
        sample_pos2 = middle_idxs[1] + 2

        maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
        maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
    else:
        filter = np.ones((5, sample_len)) / (5 * sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[1]
        sample_pos2 = middle_idxs[0] + 2

        maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
        maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
    return maze_layout


def sample_layout(seed=None,
                  size=40,
                  max_len_frac=0.5,
                  min_len_frac=0.3,
                  coverage_frac=0.25,
                  temp=20):
    rng = np.random.default_rng(seed=seed)
    maze_layout = np.zeros((size, size))

    while np.mean(maze_layout) < coverage_frac:
        maze_layout = place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp)

    return maze_layout


maze_layout = sample_layout()


RENDER_SCALE = 10
render_maze_layout = maze_layout.repeat(RENDER_SCALE, axis=0).repeat(RENDER_SCALE, axis=1)

cv2.imwrite("toy_maze.png", (1 - render_maze_layout)*255)
