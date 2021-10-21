import numpy as np

from d4rl.kitchen_2.adept_envs.franka.robot.franka_robot import Robot_VelAct
from robosuite.controllers.controller_factory import load_controller_config, controller_factory
import robosuite.utils.transform_utils as T


class Robot_RobosuiteControl(Robot_VelAct):
    """Provides interface for using robosuite controllers with Franka robot."""
    CONTROLLER = None       # should be set by subclasses to determine controller type

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = None

    def step(self, env, ctrl_desired, step_duration, sim_override=False):
        """ctrl_desired is assumed to be desired delta in EEF position and orientation (6-dim). + 1 dim for gripper."""
        if env.initializing:
            # during initialization mujoco_env will send in 9-dim action once
            ctrl_desired = ctrl_desired[:7]
        assert ctrl_desired.shape[0] == 7       # 6-dim target pose, 1-dim gripper open/close

        if self._controller is None:
            self.make_controller(env.sim)

        # get arm joint torques
        torques = self._run_controller(ctrl_desired[:-1], env.sim)

        # get gripper actuation
        if ctrl_desired[-1] == 0:
            # close gripper
            gripper_target_pos = self.robot_pos_bound[-2:, 0]
        elif ctrl_desired[-1] == 1:
            # open gripper
            gripper_target_pos = self.robot_pos_bound[-2:, 1]
        else:
            raise ValueError("Gripper actuation needs to be 0 (closed) or 1 (opened).")

        # combine output actions
        actions = np.concatenate((torques, gripper_target_pos))

        return super().step(env, actions, step_duration, sim_override)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._controller = None

    def make_controller(self, sim):
        config = self._get_controller_config(sim)
        self._controller = controller_factory(config["type"], config)
        base_pos = sim.data.get_body_xpos("panda0_link0")
        base_ori = T.mat2quat(sim.data.get_body_xmat("panda0_link0").reshape((3, 3)))
        self._controller.update_base_pose(base_pos, base_ori)

    def _get_controller_config(self, sim):
        config = load_controller_config(default_controller=self.CONTROLLER)
        config["robot_name"] = "franka"
        config["sim"] = sim
        config["eef_name"] = "end_effector"
        config["joint_indexes"] = {
            "joints": np.arange(7),
            "qpos": np.arange(7),
            "qvel": np.arange(7),
        }
        config["policy_freq"] = 30
        config["ndim"] = 7
        config["actuator_range"] = self.torque_limits(sim)
        return config

    def _run_controller(self, target_pose, sim):
        self._controller.set_goal(target_pose)
        torques = self._controller.run_controller()
        return np.clip(torques, *self.torque_limits(sim))

    def ctrl_velocity_limits(self, ctrl_desired, step_duration):
        """Don't constrain limits here since we are performing torque control."""
        return ctrl_desired

    def ctrl_position_limits(self, ctrl_desired):
        """Don't constrain limits here since we are performing torque control."""
        return ctrl_desired

    def torque_limits(self, sim):
        # Torque limit values pulled from relevant robot.xml file
        low = sim.model.actuator_ctrlrange[np.arange(7), 0]
        high = sim.model.actuator_ctrlrange[np.arange(7), 1]
        return low, high


class Robot_OSPControl(Robot_RobosuiteControl):
    """Wraps robot control in operation space controller that maps from EEF pose deltas to torques."""
    CONTROLLER = "OSC_POSE"


class Robot_IKControl(Robot_RobosuiteControl):
    """Wraps robot control in inverse kinematics controller that maps from EEF pose deltas to joint velocities."""
    CONTROLLER = "IK_POSE"

    def _get_controller_config(self, sim):
        config = super()._get_controller_config(sim)
        config['eef_rot_offset'] = np.asarray([0., 0., -0.38268343, 0.9238795]) #[0., 0., 0., 1.])  # EULER equivalent: [0, 0, -pi/4]
        config["robot_name"] = "Panda"
        return config

