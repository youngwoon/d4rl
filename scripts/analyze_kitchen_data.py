import argparse
import d4rl
import gym
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

START_POS = [27., 4.]
TARGET_POS = [18., 8.]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-mixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset()
    dataset = env.get_dataset()

    # split dataset into sequences
    seq_end_idxs = np.where(dataset['terminals'])[0]
    start = 0
    seqs = []
    for end_idx in seq_end_idxs:
        seqs.append(dict(
            states=dataset['observations'][start:end_idx + 1], #, :30],
            actions=dataset['actions'][start:end_idx + 1],
        ))
        start = end_idx + 1


    # obj2goal_matrix = -1 * np.ones((len(seqs), 7))
    # for k, seq in tqdm.tqdm(enumerate(seqs)):
    #     for i, obs in enumerate(seq['states']):
    #         obj_at_goal = env.obj_at_goal(obs)
    #         for j, obj in enumerate(obj_at_goal):
    #             if obj and obj2goal_matrix[k, j] < 0:
    #                 obj2goal_matrix[k, j] = i
    #
    # print(obj2goal_matrix)
    # np.save("kitchen_analysis_data_partialData.npy", obj2goal_matrix)

    obj2goal_matrix = np.load("kitchen_analysis_data.npy")
    # obj2goal_matrix2 = np.load("kitchen_analysis_data_partialData.npy")
    #
    first_objects = []
    second_object = []
    third_object = []
    fourth_object = []
    for row in obj2goal_matrix:
        row[row < 0] = np.infty
        first_objects.append(np.argmin(row))
        if first_objects[-1] == 0:
            row[0] = np.infty
            second_object.append(np.argmin(row))
            if second_object[-1] == 1:
                row[1] = np.infty
                third_object.append(np.argmin(row))
                # if third_object[-1] == 6:
                #     row[6] = np.infty
                #     fourth_object.append(np.argmin(row))

    plt.hist(third_object)
    plt.savefig("hist.png")
    plt.show()

    def compute_id(row):
        # first = np.argmin(row)
        # row[first] = np.infty
        # second = np.argmin(row)
        # id = int(first * 10 + second)
        # return id
        return np.argmin(row)

    # count number of occurances
    counts = defaultdict(lambda: 0.)
    for row in obj2goal_matrix.copy():
        id = compute_id(row)
        counts[id] += 1

    # compute reweighting weights
    for id in counts:
        counts[id] = counts[id]/obj2goal_matrix.shape[0]

    # compute resampling weights
    uniform_prob = 1 / len(counts.keys())
    for id in counts:
        counts[id] = uniform_prob / counts[id]

    # map probabilities back to sequences
    reweight_probs = []
    for row in obj2goal_matrix.copy():
        id = compute_id(row)
        reweight_probs.append(counts[id])

    # print(list(np.round(reweight_probs, decimals=2)))


