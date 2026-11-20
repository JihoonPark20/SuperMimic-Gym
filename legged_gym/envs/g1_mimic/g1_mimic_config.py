"""Unitree G1 whole-body motion tracking with SuperMimic (world model + imagined actor-critic).

MDP and learner settings follow supermimic-m5 @ v0.1.0 (MIT): LAFAN1, anchor-relative goal,
second-stage reset sampler. Robot,
actuators and solver follow mjlab's G1 (Unitree RL Mjlab @ 1425b15f, Apache-2.0).
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_config import BaseConfig


def _reflected_inertia(rotors, gears):
    return rotors[0] * (gears[1] * gears[2]) ** 2 + rotors[1] * gears[2] ** 2 + rotors[2]


_ARM_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ARM_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
_ARM_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARM_4010 = _reflected_inertia((0.068e-4, 0, 0), (1, 5, 5))
_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz natural frequency, damping ratio 2
# Pattern, reflected inertia, effort limit. Ankle/waist linkages use two 5020 motors.
_MOTORS = [
    (r".*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint", _ARM_5020, 25.0),
    (r"(.*_hip_pitch|.*_hip_yaw|waist_yaw)_joint", _ARM_7520_14, 88.0),
    (r".*_(hip_roll|knee)_joint", _ARM_7520_22, 139.0),
    (r".*_wrist_(pitch|yaw)_joint", _ARM_4010, 5.0),
    (r"(waist_pitch|waist_roll|.*_ankle_pitch|.*_ankle_roll)_joint", _ARM_5020 * 2, 50.0),
]


class G1MimicCfg(BaseConfig):
    seed = 0

    class env:
        num_envs = 1024
        episode_length_s = 10.0
        history_length = 10

    class sim:
        dt = 0.005  # 4 physics steps per 50 Hz control step
        decimation = 4
        nthread = 10  # MuJoCo CPU workers
        iterations = 10
        cone = "pyramidal"
        ls_iterations = 20
        tolerance = 1e-08
        ls_tolerance = 0.01
        ccd_iterations = 50

    class asset:
        file = "g1/g1_mjlab.xml"
        root_body = "pelvis"
        mass_body = "torso_link"
        joint_names = (
            [
                f"{s}_{j}_joint"
                for s in ("left", "right")
                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
            ]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [
                f"{s}_{j}_joint"
                for s in ("left", "right")
                for j in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
        )
        foot_geoms = [f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)]
        soft_joint_pos_limit = 0.9  # fraction of the joint range used by the limit penalty and resets

    class control:
        stiffness = {pattern: arm * _FREQ**2 for pattern, arm, effort in _MOTORS}
        damping = {pattern: 4 * arm * _FREQ for pattern, arm, effort in _MOTORS}
        armature = {pattern: arm for pattern, arm, effort in _MOTORS}
        effort_limits = {pattern: effort for pattern, arm, effort in _MOTORS}
        action_scale = {pattern: 0.25 * effort / (arm * _FREQ**2) for pattern, arm, effort in _MOTORS}

    class init_state:
        pos = [0, 0, 0.76]
        default_joint_angles = {  # mjlab G1 KNEES_BENT keyframe; the policy observes q - q0.
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        }

    class motion:
        npz_dir = f"{LEGGED_GYM_ROOT_DIR}/data/motions/lafan1"
        clips = None  # None selects every NPZ in npz_dir
        tracked_bodies = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        anchor_body = "torso_link"
        future_frames = 10
        anchor_obs = True  # 9 anchor-relative goal features (reference torso pose in the robot torso frame)
        action_noise_std = 0.1  # executed only; observations and rewards see the clean action
        reset_noise = True
        pose_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.01, 0.01], [-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]]
        velocity_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.2, 0.2], [-0.52, 0.52], [-0.52, 0.52], [-0.78, 0.78]]
        joint_position_range = [-0.1, 0.1]
        # Return-based reset sampling over 1 s bins (start | uniform | return).
        sampling_mode = "return"
        bin_frames = 50
        uniform_ratio = 0.8
        hard_ratio = 0.15
        memory_ratio = 0.05
        softmax_temp = 5.0
        g_max = 10.0
        gamma = 0.99
        ema_alpha = 0.05
        memory_alpha = 0.01
        min_remaining_steps = 100
        record_transitions = True
        replay_capacity = 1000  # rows per environment

    class noise:
        add_noise = True
        joint_pos = 0.01
        joint_vel = 0.5
        ang_vel = 0.2
        gravity = 0.05
        anchor_pos = 0.25
        anchor_ori = 0.05

    class domain_rand:
        randomize_friction = True  # foot friction, resampled every episode; 1.0 when disabled
        friction_range = [0.3, 1.2]
        randomize_com = True  # torso COM offset per axis, at startup
        com_range = [[-0.025, -0.05, -0.05], [0.025, 0.05, 0.05]]
        randomize_encoder_bias = True  # resampled every episode
        encoder_bias_range = [-0.01, 0.01]

    class termination:
        root_height = 0.25  # |reference - robot| pelvis height, m
        root_tilt = 0.8  # |reference - robot| projected-gravity z


class G1MimicCfgSuperMimic(BaseConfig):
    seed = 0

    class runner:
        experiment_name = "g1_mimic"
        max_iterations = 21000  # environment steps in this invocation (additional when resuming)
        run_name = ""
        save_interval = 500
        log_interval = 50

    class supermimic:
        total_iterations = 320000  # learning-rate schedule horizon in environment steps
        updates_per_env_step = 0.4
        warmup_iterations = 40  # collection steps before the first update, also after resume
        learning_rate = 2e-4
        lr_final_ratio = 0.03
        lr_decay_fraction = 0.78125
        max_hours = None

    class world_model:
        hidden = [2048, 2048, 2048]
        term_hidden = [64, 64]
        window = 8
        batches_per_update = 10
        batch_size = 512
        grad_clip = 1.0
        compile = "default"  # torch.compile mode or None

    class policy:
        actor = "mlp"
        ode_dmax = None
        ode_amax = None
        encoder_hidden = [2048, 1024, 512, 256]
        decoder_hidden = [2048, 1024, 512, 256]
        latent_dim = 32
        decoder_last_gain = 0.01

    class critic:
        hidden = [256, 128]

    class imagination:
        horizon = 32
        batch_size = 512
        gamma = 0.99
        gae_lambda = 0.95
        betas = [0.9, 0.95]
        grad_clip_norm = 1.0
        grad_clip_value = 0.1
        critic_grad_clip = 1.0
        critic_epochs = 5
        critic_minibatches = 4
        obs_noise = True
        compile = "default"


class G1MimicCfgSuperMimicODE(G1MimicCfgSuperMimic):
    """Same MDP and learner as SuperMimic; only the actor's action parameterization changes."""

    class policy(G1MimicCfgSuperMimic.policy):
        actor = "ode"
        ode_dmax = 1.62
        # NTUMARS/ODE-POLICY @ 9c505796, rl/supermimic_ode/ode_amax_g1.txt.
        # Normalized action units, G1 policy joint order: left leg, right leg,
        # waist, left arm, right arm. Empirical bounds, not physical joint limits.
        ode_amax = (
            4.84,
            4.71,
            3.34,
            7.22,
            5.91,
            1.58,
            5.0,
            3.52,
            3.32,
            6.94,
            5.7,
            1.53,
            2.13,
            2.78,
            3.33,
            6.16,
            5.29,
            4.68,
            4.73,
            5.11,
            15.52,
            22.56,
            6.11,
            5.22,
            4.84,
            4.71,
            4.91,
            14.55,
            21.26,
        )
