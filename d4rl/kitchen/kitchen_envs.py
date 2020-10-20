"""Environments using kitchen and Franka robot."""
import os
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

from d4rl.offline_env import OfflineEnv

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3

@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        super(KitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score)

    def _get_task_goal(self, task=None):
        if task is None:
            task = ['microwave', 'kettle', 'bottom burner', 'light switch']
        new_goal = np.zeros_like(self.goal)
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = self._get_task_goal(task=self.TASK_ELEMENTS) #obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        env_info['completed_tasks'] = set(self.TASK_ELEMENTS) - set(self.tasks_to_complete)
        return obs, reward, done, env_info

    # def render(self, mode='human'):
    #     # Disable rendering to speed up environment evaluation.
    #     return []

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def set_state(self, qpos):
        self.robot.reset(self, qpos, np.zeros_like(self.init_qvel))
        self.sim.forward()

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data['terminals'])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(dict(
                states=data['observations'][start:end_idx + 1],
                actions=data['actions'][start:end_idx + 1],
            ))
            start = end_idx + 1
        return seqs


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

    def get_goal(self):
        data = self.get_dataset()
        seqs = self._split_data_into_seqs(data)
        return seqs[0]['states'][-1]


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    # TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'light switch']
    #TASK_ELEMENTS = ['bottom burner', 'microwave', 'kettle', 'light switch']
    #TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'light switch']
    #TASK_ELEMENTS = ['bottom burner']
    #TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet']
    #TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Kitchen Env task is modified!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    TASK_ELEMENTS = ['microwave', 'kettle', 'hinge cabinet', 'light switch']
