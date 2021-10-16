import numpy as np

from d4rl.kitchen_2.adept_envs.franka.robot.franka_robot import Robot_VelAct
from robosuite.controllers.controller_factory import load_controller_config, controller_factory


class Robot_OSPControl(Robot_VelAct):
    """Wraps robot control in operation space controller that maps from EEF pose deltas to joint velocities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = None

    def step(self, env, ctrl_desired, step_duration, sim_override=False):
        if self._controller is None:
            self.make_controller(env.sim)

        actions = self._controller.control(ctrl_desired, step_duration)
        return super().step(env, actions, step_duration, sim_override)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._controller = None

    def make_controller(self, sim):
        config = load_controller_config(default_controller="OSC_POSE")
        config["robot_name"] = "franka"
        config["sim"] = sim
        config["eef_name"] = "end_effector"
        config["joint_indexes"] = {
            "joints": np.arrange(7),
            "qpos": np.arrange(7),
            "qvel": np.arrange(7),
        }
        config["policy_freq"] = 30
        config["ndim"] = 7

