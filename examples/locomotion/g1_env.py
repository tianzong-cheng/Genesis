import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class G1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.history_length = obs_cfg.get("history_length", 1)
        self.num_actions = env_cfg["num_actions"]
        self.num_dof = env_cfg["num_dof"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.0, 3.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=50,
                enable_self_collision=False,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=env_cfg.get("urdf_path", "urdf/g1/g1_29dof.urdf"),
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # add camera
        if self.env_cfg.get("record", False):
            self.record_camera = self.scene.add_camera(
                res=(640, 480),
                pos=(5.0, 5.0, 1.0),
                lookat=(0, 0, 0.5),
                fov=50,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp(self.env_cfg["kp"], self.motors_dof_idx)
        self.robot.set_dofs_kv(self.env_cfg["kd"], self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs * self.history_length), device=gs.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["height"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_dof), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=gs.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        # Initialize circular buffer for observation history
        # Buffer shape: (num_envs, history_length, num_obs)
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.history_length, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        self.history_idx = 0  # Current position in circular buffer

        # Upper body action buffers
        self.upper_actions = torch.zeros(
            (self.num_envs, self.env_cfg["num_upper_dof"]), device=gs.device, dtype=gs.tc_float
        )
        self.step_upper_target_delta = torch.zeros_like(self.upper_actions)
        self.upper_resample_interval_s = self.env_cfg.get("upper_resample_interval_s", 1.0)

        # Obtain joint limits
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        for joint in self.robot.joints:
            lower = joint.dofs_limit[0][0]
            upper = joint.dofs_limit[0][1]
            self.joint_lower_limits.append(lower)
            self.joint_upper_limits.append(upper)
        self.joint_lower_limits = torch.tensor(self.joint_lower_limits, device=gs.device, dtype=gs.tc_float)
        self.joint_upper_limits = torch.tensor(self.joint_upper_limits, device=gs.device, dtype=gs.tc_float)
        waist_roll_pitch_idx = [5, 8]  # waist_roll_joint, waist_pitch_joint
        self.joint_lower_limits[waist_roll_pitch_idx] = 0.0
        self.joint_upper_limits[waist_roll_pitch_idx] = 0.0

        self.step_count = 0

    def _resample_commands(self, envs_idx):
        # task selection
        height_task_prob = self.command_cfg.get("height_task_prob", 0.0)
        task_selector = gs_rand_float(0.0, 1.0, (len(envs_idx),), gs.device)
        height_task_idx = (task_selector < height_task_prob).nonzero(as_tuple=False).reshape(-1)
        velocity_task_idx = (task_selector >= height_task_prob).nonzero(as_tuple=False).reshape(-1)

        # velocity task
        self.commands[velocity_task_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (len(velocity_task_idx),), gs.device
        )
        self.commands[velocity_task_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (len(velocity_task_idx),), gs.device
        )
        self.commands[velocity_task_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"], (len(velocity_task_idx),), gs.device
        )
        self.commands[velocity_task_idx, 3] = self.command_cfg["height_range"][1]  # 0.74m

        # height task
        self.commands[height_task_idx, :3] = 0.0
        self.commands[height_task_idx, 3] = gs_rand_float(
            *self.command_cfg["height_range"], (len(height_task_idx),), gs.device
        )

    def compute_upper_actions(self):
        self.step_count += 1
        resample_interval_steps = int(self.upper_resample_interval_s / self.dt)
        if (self.step_count % resample_interval_steps) == 0:
            uniform_samples = torch.rand(
                self.num_envs, self.env_cfg["num_upper_dof"], device=gs.device, dtype=gs.tc_float
            )
            joint_range = self.joint_upper_limits - self.joint_lower_limits
            self.interval_upper_target = (
                self.joint_lower_limits[self.env_cfg["upper_dof"]]
                + uniform_samples * joint_range[self.env_cfg["upper_dof"]]
            )
            self.step_upper_target_delta = (self.interval_upper_target - self.upper_actions) / resample_interval_steps
        self.upper_actions += self.step_upper_target_delta

    def step(self, actions):
        # Compute upper body actions
        self.compute_upper_actions()
        whole_body_actions = torch.zeros(self.num_envs, self.num_dof, device=gs.device, dtype=gs.tc_float)
        whole_body_actions[:, self.env_cfg["lower_dof"]] = actions
        whole_body_actions[:, self.env_cfg["upper_dof"]] = self.upper_actions
        self.actions = whole_body_actions

        clipped_actions = torch.clip(whole_body_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = clipped_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        if len(envs_idx) > 0:
            self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        current_obs = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 4
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # num_dof
                self.dof_vel * self.obs_scales["dof_vel"],  # num_dof
                self.actions,  # num_dof
            ],
            axis=-1,
        )

        # Update circular buffer with new observation
        self.obs_history_buf[:, self.history_idx] = current_obs
        self.history_idx = (self.history_idx + 1) % self.history_length

        # Return flattened history buffer as actor observation
        # Roll buffer so oldest observations come first, newest last (FIFO order)
        ordered_buffer = torch.roll(self.obs_history_buf, shifts=self.history_length - self.history_idx, dims=1)
        self.obs_buf = ordered_buffer.reshape(self.num_envs, -1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = torch.zeros(
            (self.num_envs, (self.num_obs + 5) * self.obs_cfg["history_length"]), device=gs.device, dtype=gs.tc_float
        )  # Critic observation is not used since this is a sim-to-sim experiment

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = torch.zeros(
            (self.num_envs, (self.num_obs + 5) * self.obs_cfg["history_length"]), device=gs.device, dtype=gs.tc_float
        )  # Critic observation is not used since this is a sim-to-sim experiment
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # reset observation history buffer
        self.obs_history_buf[envs_idx] = 0.0

        # reset upper body action buffers
        self.upper_actions[envs_idx] = 0.0
        self.step_upper_target_delta[envs_idx] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
