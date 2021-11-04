"""Environments using kitchen and Franka robot."""
import os
import numpy as np
from d4rl.kitchen_2.adept_envs.utils.configurable import configurable
from d4rl.kitchen_2.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

from d4rl.offline_env import OfflineEnv

SEMANTIC_SKILLS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet',
                   'hinge cabinet', 'microwave', 'kettle']
OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([9, 10]), # (from 11, 12)
    'top burner': np.array([13, 14]), # (from 15, 16)
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
    'hinge cabinet': np.array([0, 1.2]),  #-1.2, 0]), # (from np.array([0., 1.45]),)
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.27, 0.75, 1.62, 0.99, 0., 0., -0.06]), # from [-0.23, ...]
    }
GOAL_APPROACH_SITES = {
    'bottom burner': "knob1_site",
    'top burner': "knob3_site",
    'light switch': "light_site",
    'slide cabinet': "slide_site",
    'hinge cabinet': "hinge_site2", #1",
    'microwave': "microhandle_site",
    'kettle': "kettle_site",
}
BONUS_THRESH = 0.25

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
    ENFORCE_TASK_ORDER = True

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
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]

            # compute distance to goal
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])

            # check whether we completed the task
            complete = distance < BONUS_THRESH * np.linalg.norm(self.init_qpos[element_idx] - next_goal[element_idx])
            if complete and (all_completed_so_far or not self.ENFORCE_TASK_ORDER):
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus

        if self.DENSE_REWARD:
            reward_dict['goal_dist'] = np.linalg.norm(obs_dict['goal_dist'])
            reward_dict['approach_dist'] = np.linalg.norm(obs_dict['approach_dist'])
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
        return obs, reward, done, env_info

    # def render(self, mode='human'):
    #     # Disable rendering to speed up environment evaluation.
    #     return []

    def _get_obs(self):
        obs = super()._get_obs()
        if self.DENSE_REWARD:
            assert len(self.tasks_to_complete) == 1, "Currently only supports single-object skills for dense rewards."

            # compute distance to next subgoal
            element_idx = OBS_ELEMENT_INDICES[self.tasks_to_complete[0]]
            self.obs_dict['goal_dist'] = self.obs_dict['obj_qp'][..., element_idx - len(self.obs_dict['qp'])] \
                    - self._get_task_goal(task=self.TASK_ELEMENTS)[element_idx]

            # compute distance to next approach site
            self.obs_dict['approach_dist'] = \
                self.sim.data.site_xpos[self.sim.model.site_name2id(GOAL_APPROACH_SITES[self.tasks_to_complete[0]])] \
                    - self.sim.data.site_xpos[self.sim.model.site_name2id("end_effector")]

            # concatenate goal and approach dists to observation
            # obs = np.concatenate([obs, self.obs_dict['goal_dist'], self.obs_dict['approach_dist']])
        return obs

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

    @staticmethod
    def check_task_done(task_name, obs):
        assert task_name in OBS_ELEMENT_INDICES, "Task {} is not defined!".format(task_name)
        return np.linalg.norm(obs[..., OBS_ELEMENT_INDICES[task_name]] - OBS_ELEMENT_GOALS[task_name]) < BONUS_THRESH

    @staticmethod
    def semantic_skills():
        return SEMANTIC_SKILLS


class KitchenRand(KitchenBase):
    """Randomly initializes agent + environment (except for target objects)."""
    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()

        # randomly sample joints & gripper position
        joint_limits = self.robot.robot_pos_bound[:self.n_jnt]   # [n_joints, 2]
        for i, joint_limit in enumerate(joint_limits):
            reset_pos[i] = np.random.uniform(joint_limit[0], joint_limit[1])
        reset_pos[8] = reset_pos[7]     # ensure that initial gripper opening is symmetric

        # for each object that is not target object: sample in range(start, goal)
        for object in OBS_ELEMENT_INDICES:
            if object in self.TASK_ELEMENTS:
                # don't change initial position of task goal objects
                continue
            for obj_element_idx, obj_element_goal in zip(OBS_ELEMENT_INDICES[object], OBS_ELEMENT_GOALS[object]):
                reset_pos[obj_element_idx] = np.random.uniform(min(reset_pos[obj_element_idx], obj_element_goal),
                                                               max(reset_pos[obj_element_idx], obj_element_goal))

        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return self._get_obs()


class KitchenDatasetInit(KitchenBase):
    """Loads states from given rollout sequences & initializes environment accordingly."""
    DATASET_PATH = "/Users/kpertsch/data/kitchen_2_rollouts"        # path to root folder for datasets


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load dataset states into memory
        self._init_states = {}
        for task in OBS_ELEMENT_INDICES:
            self._init_states[task] = self._load_states(os.path.join(self.DATASET_PATH, task.replace(' ', '_')))

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()

        # randomly sample joints & gripper position from the loaded dataset
        sample_task = np.random.choice(list(OBS_ELEMENT_INDICES))
        sample_state = self._init_states[sample_task][np.random.randint(self._init_states[sample_task].shape[0])]
        reset_pos[:9] = sample_state[:9]

        # for each object that is not target object and not sampled task: sample in range(start, goal)
        for object in OBS_ELEMENT_INDICES:
            if object in self.TASK_ELEMENTS or object == sample_task:
                # don't change initial position of task goal objects
                continue
            for obj_element_idx, obj_element_goal in zip(OBS_ELEMENT_INDICES[object], OBS_ELEMENT_GOALS[object]):
                reset_pos[obj_element_idx] = np.random.uniform(min(reset_pos[obj_element_idx], obj_element_goal),
                                                               max(reset_pos[obj_element_idx], obj_element_goal))

        # set object from sampled task to position from sampled state
        reset_pos[OBS_ELEMENT_INDICES[sample_task]] = sample_state[OBS_ELEMENT_INDICES[sample_task]]

        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return self._get_obs()

    def _load_states(self, data_dir):
        # load all H5 filenames in directory
        filenames = []
        for root, dirs, files in os.walk(data_dir, followlinks=True):
            for file in files:
                if file.endswith(".h5"): filenames.append(os.path.join(root, file))
        if not filenames:
            raise ValueError("Did not find rollouts in {}!".format(data_dir))

        # load states from trajectories
        import tqdm, h5py
        states = []
        for traj in tqdm.tqdm(filenames):
            with h5py.File(traj, 'r') as F:
                states.append(F['traj0/states'][()])
        return np.concatenate(states)


class KitchenAllTasksV0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet',
                     'hinge cabinet', 'microwave', 'kettle']
    ENFORCE_TASK_ORDER = False


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


# MULTI TASK -- DATA COLLECT -- FIXED ENVS

class Kitchen_BB_TB_LS_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet']

class Kitchen_BB_TB_SC_LS_V0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'slide cabinet', 'light switch']

class Kitchen_BB_TB_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'slide cabinet', 'hinge cabinet']

class Kitchen_MW_BB_TB_LS_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'top burner', 'light switch']

class Kitchen_MW_BB_TB_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'top burner', 'slide cabinet']

class Kitchen_MW_BB_TB_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'top burner', 'hinge cabinet']

class Kitchen_MW_BB_LS_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'light switch', 'top burner']

class Kitchen_MW_BB_LS_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'light switch', 'slide cabinet']

class Kitchen_MW_BB_SC_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'slide cabinet', 'top burner']

class Kitchen_MW_BB_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'slide cabinet', 'hinge cabinet']

class Kitchen_MW_BB_HC_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'hinge cabinet', 'top burner']

class Kitchen_MW_TB_LS_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'top burner', 'light switch', 'hinge cabinet']

class Kitchen_MW_LS_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'light switch', 'slide cabinet', 'hinge cabinet']

class Kitchen_MW_KET_BB_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'top burner']

class Kitchen_MW_KET_BB_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'slide cabinet']

class Kitchen_MW_KET_BB_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'hinge cabinet']

class Kitchen_MW_KET_TB_LS_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'top burner', 'light switch']

class Kitchen_MW_KET_TB_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'top burner', 'hinge cabinet']

class Kitchen_MW_KET_LS_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

class Kitchen_MW_KET_LS_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'hinge cabinet']

class Kitchen_MW_KET_SC_BB_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'bottom burner']

class Kitchen_MW_KET_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet']

class Kitchen_KET_BB_TB_LS_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'top burner', 'light switch']

class Kitchen_KET_BB_TB_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'top burner', 'slide cabinet']

class Kitchen_KET_BB_TB_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'top burner', 'hinge cabinet']

class Kitchen_KET_BB_LS_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'light switch', 'top burner']

class Kitchen_KET_BB_LS_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'light switch', 'slide cabinet']

class Kitchen_KET_BB_LS_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'light switch', 'hinge cabinet']

class Kitchen_KET_BB_SC_TB_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'slide cabinet', 'top burner']

class Kitchen_KET_BB_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'bottom burner', 'slide cabinet', 'hinge cabinet']

class Kitchen_KET_TB_LS_SC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'top burner', 'light switch', 'slide cabinet']

class Kitchen_KET_LS_SC_BB_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'light switch', 'slide cabinet', 'bottom burner']

class Kitchen_KET_LS_SC_HC_V0(KitchenBase):
    TASK_ELEMENTS = ['kettle', 'light switch', 'slide cabinet', 'hinge cabinet']


# SINGLE TASK -- FIXED ENVS

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


# SINGLE TASK -- RANDOMIZED START ENVS

class KitchenBottomBurnerRandV0(KitchenRand):
    TASK_ELEMENTS = ['bottom burner']
    DENSE_REWARD = True


class KitchenTopBurnerRandV0(KitchenRand):
    TASK_ELEMENTS = ['top burner']
    DENSE_REWARD = True


class KitchenLightSwitchRandV0(KitchenRand):
    TASK_ELEMENTS = ['light switch']
    DENSE_REWARD = True


class KitchenSlideCabinetRandV0(KitchenRand):
    TASK_ELEMENTS = ['slide cabinet']
    DENSE_REWARD = True


class KitchenHingeCabinetRandV0(KitchenRand):
    TASK_ELEMENTS = ['hinge cabinet']
    DENSE_REWARD = True


class KitchenMicrowaveRandV0(KitchenRand):
    TASK_ELEMENTS = ['microwave']
    DENSE_REWARD = True


class KitchenKettleRandV0(KitchenRand):
    TASK_ELEMENTS = ['kettle']
    DENSE_REWARD = True


# SINGLE TASK -- DATALOADED INIT ENVS

class KitchenBottomBurnerDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['bottom burner']
    DENSE_REWARD = True


class KitchenTopBurnerDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['top burner']
    DENSE_REWARD = True


class KitchenLightSwitchDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['light switch']
    DENSE_REWARD = True


class KitchenSlideCabinetDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['slide cabinet']
    DENSE_REWARD = True


class KitchenHingeCabinetDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['hinge cabinet']
    DENSE_REWARD = True


class KitchenMicrowaveDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['microwave']
    DENSE_REWARD = True


class KitchenKettleDataInitV0(KitchenDatasetInit):
    TASK_ELEMENTS = ['kettle']
    DENSE_REWARD = True
