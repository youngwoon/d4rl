# from .kitchen_envs import KitchenMicrowaveKettleLightSliderV0, KitchenMicrowaveKettleBottomBurnerLightV0, \
#                              KitchenKettleBottomBurnerTopBurnerSliderV0, KitchenMicrowaveLightSliderHingeV0
from .kitchen_envs import *
from gym.envs.registration import register

# Smaller dataset with only positive demonstrations.
register(
    id='kitchen-complete-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. A subset of the demonstrations
# solve the task.
register(
    id='kitchen-partial-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id='kitchen-mixed-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

####### SkiLD Tasks #########
register(
    id='kitchen-kbts-v0',
    entry_point='d4rl.kitchen:KitchenKettleBottomBurnerTopBurnerSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-mlsh-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveLightSliderHingeV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)
#############################

### SINGLE TASK -- FIXED envs

register(
    id='kitchen-BB-v0',
    entry_point='d4rl.kitchen:KitchenBottomBurnerV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-TB-v0',
    entry_point='d4rl.kitchen:KitchenTopBurnerV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-LS-v0',
    entry_point='d4rl.kitchen:KitchenLightSwitchV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-SC-v0',
    entry_point='d4rl.kitchen:KitchenSlideCabinetV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-HC-v0',
    entry_point='d4rl.kitchen:KitchenHingeCabinetV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-MW-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-KET-v0',
    entry_point='d4rl.kitchen:KitchenKettleV0',
    max_episode_steps=50,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
