def g1_env_cfgs():
    env_cfg = {
        "num_actions": 12,
        "num_dof": 29,
        # joint/link names
        "default_joint_angles": {
            "left_hip_pitch_joint": -0.20,
            "right_hip_pitch_joint": -0.20,
            "waist_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_knee_joint": 0.42,
            "right_knee_joint": 0.42,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": 0.35,
            "left_ankle_pitch_joint": -0.23,
            "right_ankle_pitch_joint": -0.23,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.87,
            "right_elbow_joint": 0.87,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },  # [rad]
        "joint_names": [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        ],
        # PD
        "kp": [
            200.0,  # left_hip_pitch_joint
            200.0,  # right_hip_pitch_joint
            200.0,  # waist_yaw_joint
            150.0,  # left_hip_roll_joint
            150.0,  # right_hip_roll_joint
            200.0,  # waist_roll_joint
            150.0,  # left_hip_yaw_joint
            150.0,  # right_hip_yaw_joint
            200.0,  # waist_pitch_joint
            200.0,  # left_knee_joint
            200.0,  # right_knee_joint
            100.0,  # left_shoulder_pitch_joint
            100.0,  # right_shoulder_pitch_joint
            20.0,  # left_ankle_pitch_joint
            20.0,  # right_ankle_pitch_joint
            100.0,  # left_shoulder_roll_joint
            100.0,  # right_shoulder_roll_joint
            20.0,  # left_ankle_roll_joint
            20.0,  # right_ankle_roll_joint
            50.0,  # left_shoulder_yaw_joint
            50.0,  # right_shoulder_yaw_joint
            50.0,  # left_elbow_joint
            50.0,  # right_elbow_joint
            40.0,  # left_wrist_roll_joint
            40.0,  # right_wrist_roll_joint
            40.0,  # left_wrist_pitch_joint
            40.0,  # right_wrist_pitch_joint
            40.0,  # left_wrist_yaw_joint
            40.0,  # right_wrist_yaw_joint
        ],
        "kd": [
            5.0,  # left_hip_pitch_joint
            5.0,  # right_hip_pitch_joint
            5.0,  # waist_yaw_joint
            5.0,  # left_hip_roll_joint
            5.0,  # right_hip_roll_joint
            5.0,  # waist_roll_joint
            5.0,  # left_hip_yaw_joint
            5.0,  # right_hip_yaw_joint
            5.0,  # waist_pitch_joint
            5.0,  # left_knee_joint
            5.0,  # right_knee_joint
            2.0,  # left_shoulder_pitch_joint
            2.0,  # right_shoulder_pitch_joint
            2.0,  # left_ankle_pitch_joint
            2.0,  # right_ankle_pitch_joint
            2.0,  # left_shoulder_roll_joint
            2.0,  # right_shoulder_roll_joint
            2.0,  # left_ankle_roll_joint
            2.0,  # right_ankle_roll_joint
            2.0,  # left_shoulder_yaw_joint
            2.0,  # right_shoulder_yaw_joint
            2.0,  # left_elbow_joint
            2.0,  # right_elbow_joint
            2.0,  # left_wrist_roll_joint
            2.0,  # right_wrist_roll_joint
            2.0,  # left_wrist_pitch_joint
            2.0,  # right_wrist_pitch_joint
            2.0,  # left_wrist_yaw_joint
            2.0,  # right_wrist_yaw_joint]
        ],
        # termination
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.80],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # HOMIE related
        "lower_dof": [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18],
        "upper_dof": [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        "num_upper_dof": 17,
    }
    obs_cfg = {
        "num_obs": 97,
        "history_length": 6,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "height": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 1.0,
        },
    }
    reward_cfg = {
        "reward_scales": {},
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-0.6, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        "height_range": [0.24, 0.74],
        "height_task_prob": 0.3,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def g1_agent_cfg():
    agent_cfg = {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": 50000,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "symmetry_cfg": None,  # RslRlSymmetryCfg()
            "rnd_cfg": None,  # RslRlRndCfg()
        },
        "clip_actions": None,
        "save_interval": 100,
        "experiment_name": "",
        "run_name": "",
        "logger": "wandb",
        "neptune_project": "leggedlab",
        "wandb_project": "leggedlab",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
    }

    return agent_cfg
