from .kitchen_envs import *
from gym.envs.registration import register

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id='kitchen-2-mixed-v0',
    entry_point='d4rl.kitchen_2:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=100000, #280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)

### MULTI TASK -- DATA COLLECT FIXED envs

register(
    id='kitchen-2-BB-TB-LS-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_BB_TB_LS_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-BB-TB-SC-LS-v0',
    entry_point='d4rl.kitchen_2:Kitchen_BB_TB_SC_LS_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-BB-TB-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_BB_TB_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-TB-LS-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_TB_LS_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-TB-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_TB_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-TB-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_TB_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-LS-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_LS_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-LS-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_LS_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-SC-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_SC_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-BB-HC-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_BB_HC_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-TB-LS-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_TB_LS_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-LS-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_LS_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-BB-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_BB_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-BB-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_BB_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-BB-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_BB_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-TB-LS-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_TB_LS_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-TB-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_TB_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-LS-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_LS_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-LS-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_LS_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-SC-BB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_SC_BB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-KET-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_MW_KET_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-TB-LS-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_TB_LS_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-TB-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_TB_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-TB-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_TB_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-LS-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_LS_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-LS-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_LS_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-LS-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_LS_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-SC-TB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_SC_TB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-BB-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_BB_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-TB-LS-SC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_TB_LS_SC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-LS-SC-BB-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_LS_SC_BB_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-LS-SC-HC-v0',
    entry_point='d4rl.kitchen_2:Kitchen_KET_LS_SC_HC_V0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)


### SINGLE TASK -- FIXED envs

register(
    id='kitchen-2-BB-v0',
    entry_point='d4rl.kitchen_2:KitchenBottomBurnerV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-TB-v0',
    entry_point='d4rl.kitchen_2:KitchenTopBurnerV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-LS-v0',
    entry_point='d4rl.kitchen_2:KitchenLightSwitchV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-SC-v0',
    entry_point='d4rl.kitchen_2:KitchenSlideCabinetV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-HC-v0',
    entry_point='d4rl.kitchen_2:KitchenHingeCabinetV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-v0',
    entry_point='d4rl.kitchen_2:KitchenMicrowaveV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-v0',
    entry_point='d4rl.kitchen_2:KitchenKettleV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)


### SINGLE TASK -- RANDOMIZED START envs

register(
    id='kitchen-2-BB-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenBottomBurnerRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-TB-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenTopBurnerRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-LS-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenLightSwitchRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-SC-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenSlideCabinetRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-HC-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenHingeCabinetRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenMicrowaveRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-rand-v0',
    entry_point='d4rl.kitchen_2:KitchenKettleRandV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)


### SINGLE TASK -- DATASET LOADED START envs

register(
    id='kitchen-2-BB-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenBottomBurnerDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-TB-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenTopBurnerDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-LS-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenLightSwitchDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-SC-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenSlideCabinetDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-HC-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenHingeCabinetDataInitV0',
    max_episode_steps=70,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-MW-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenMicrowaveDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)
register(
    id='kitchen-2-KET-datainit-v0',
    entry_point='d4rl.kitchen_2:KitchenKettleDataInitV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    }
)


### ALL TASKS ###
register(
    id='kitchen-2-all-tasks-v0',
    entry_point='d4rl.kitchen_2:KitchenAllTasksV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)
