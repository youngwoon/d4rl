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
GOAL_APPROACH_SITES = {
    'bottom burner': "knob2_site",
    'top burner': "knob4_site",
    'light switch': "light_site",
    'slide cabinet': "slide_site",
    'hinge cabinet': "hinge_site2",
    'microwave': "microhandle_site",
    'kettle': "kettle_site",
}
BONUS_THRESH = 0.3

@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    DENSE_REWARD = False

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
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
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = self._get_task_goal(task=self.TASK_ELEMENTS) #obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        goal_dists, approach_dists = [], []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            # compute distance to approach site
            approach_dist = np.linalg.norm(
                self.sim.data.site_xpos[self.sim.model.site_name2id(GOAL_APPROACH_SITES[element])]
                - self.sim.data.site_xpos[self.sim.model.site_name2id("end_effector")])
            approach_dists.append(approach_dist)

            # compute distance to goal
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            goal_dists.append(distance)

            # check whether we completed the task
            complete = distance < BONUS_THRESH
            if complete and all_completed_so_far: #element == self.tasks_to_complete[0]:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus

        if self.DENSE_REWARD:
            reward_dict['goal_dist'] = np.sum(goal_dists)
            reward_dict['approach_dist'] = np.sum(approach_dists)
            reward_dict['r_total'] = -1 * reward_dict['goal_dist'] - 0.5 * reward_dict['approach_dist'] + bonus
        else:
            reward_dict['r_total'] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete: 
                    done = True
                    break
        env_info['completed_tasks'] = set(self.TASK_ELEMENTS) - set(self.tasks_to_complete)

        if self.DENSE_REWARD:
            # append goal and approach dists to observation
            obs = np.concatenate([obs, np.asarray([env_info['rewards']['goal_dist'],
                                                   env_info['rewards']['approach_dist']])])
        return obs, reward, done, env_info

    # def render(self, mode='human'):
    #     # Disable rendering to speed up environment evaluation.
    #     return []

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

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


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'light switch']


class KitchenKettleBottomBurnerTopBurnerSliderV0(KitchenBase):
    # well-aligned SkiLD task
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'top burner', 'slide cabinet']


class KitchenMicrowaveLightSliderHingeV0(KitchenBase):
    # mis-aligned SkiLD task
    TASK_ELEMENTS = ['microwave', 'light switch', 'slide cabinet', 'hinge cabinet']


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

    def get_goal(self):
        data = self.get_dataset()
        seqs = self._split_data_into_seqs(data)
        return seqs[1]['states'][-1]


class KitchenBottomBurnerV0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner']
    DENSE_REWARD = True

class KitchenTopBurnerV0(KitchenBase):
    TASK_ELEMENTS = ['top burner']
    DENSE_REWARD = True

class KitchenLightSwitchV0(KitchenBase):
    TASK_ELEMENTS = ['light switch']
    DENSE_REWARD = True

class KitchenSlideCabinetV0(KitchenBase):
    TASK_ELEMENTS = ['slide cabinet']
    DENSE_REWARD = True

class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ['hinge cabinet']
    DENSE_REWARD = True

class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ['microwave']
    DENSE_REWARD = True

class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ['kettle']
    DENSE_REWARD = True
