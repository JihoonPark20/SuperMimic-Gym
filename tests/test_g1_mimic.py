"""Checks for the SuperMimic motion tracking task: python -m pytest tests -q"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.supermimic.kinematics import G1Kinematics, load_g1_model
from legged_gym.supermimic.replay import ReplayBuffer
from legged_gym.supermimic.world_model import StateSpec, StateSpecCfg

MOTIONS = Path(LEGGED_GYM_ROOT_DIR) / "data" / "motions" / "lafan1"


def _random_state(model, rng, n):
    qpos = np.tile(model.qpos0, (n, 1))
    qpos[:, :3] = rng.normal(size=(n, 3))
    quat = rng.normal(size=(n, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 7:] += rng.uniform(-0.5, 0.5, (n, model.nq - 7))
    return qpos, rng.normal(size=(n, model.nv))


def test_fk_matches_mujoco():
    model = load_g1_model()
    data = mujoco.MjData(model)
    qpos, qvel = _random_state(model, np.random.default_rng(0), 8)
    out = G1Kinematics(model, dtype=torch.float64)(torch.from_numpy(qpos), torch.from_numpy(qvel))
    for i in range(len(qpos)):
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(out["pos"][i].numpy(), data.xpos[1:], atol=1e-9)
        for b in range(1, model.nbody):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, velocity, 0)
            np.testing.assert_allclose(out["ang_vel"][i, b - 1].numpy(), velocity[:3], atol=1e-9)
            np.testing.assert_allclose(out["lin_vel"][i, b - 1].numpy(), velocity[3:], atol=1e-9)


def test_zero_dynamics_extrapolates_constant_velocity():
    spec = StateSpec(StateSpecCfg(), "cpu")
    model = load_g1_model()
    qpos, qvel = _random_state(model, np.random.default_rng(1), 4)
    s = torch.from_numpy(np.concatenate((qpos, qvel), axis=1)).float()
    s_next = spec.integrate(s, torch.zeros(4, 70))
    torch.testing.assert_close(s_next[:, 36:], s[:, 36:])
    torch.testing.assert_close(s_next[:, :3], s[:, :3] + 0.02 * s[:, 36:39])
    assert spec.normalized_sq_error(s, s).abs().max() < 1e-10


def test_replay_windows_stay_inside_episodes():
    buffer = ReplayBuffer(num_envs=1, capacity=32, state_dim=1, act_dim=1, xi_dim=1)
    env = torch.tensor([0])
    for episode, length in enumerate((5, 9)):
        for t in range(length + 1):
            terminal = t == length
            value = torch.tensor([[10.0 * episode + t]])
            buffer.add(
                env,
                value,
                value,
                value,
                value,
                torch.zeros(1),
                torch.tensor([float(terminal)]),
                torch.tensor([t]),
                torch.tensor([terminal]),
                torch.tensor([t == 0]),
            )
        buffer.end_episode(env)
    windows = buffer.sample_windows(256, 4)
    states = windows.states[..., 0]
    assert torch.all(torch.floor(states / 10) == torch.floor(states[:1] / 10)), "a window crossed an episode"
    assert torch.all(states[1:] - states[:-1] == 1)
    start = buffer.sample_starts(256, 6)
    assert torch.all(torch.floor(start.hist_states[..., 0] / 10) == torch.floor(start.state[None, :, 0] / 10))


CHECKPOINTS = sorted((Path(LEGGED_GYM_ROOT_DIR) / "checkpoints").glob("*.pt"))
needs_motions = pytest.mark.skipif(not MOTIONS.is_dir(), reason="run python -m legged_gym.scripts.prepare_motions")


def _make(argv, evaluation=False):
    from legged_gym.envs import task_registry
    from legged_gym.supermimic.benchmark import configure_evaluation
    from legged_gym.utils import get_args

    args = get_args(["--rl_device", "cpu", "--num_threads", "2", "--motion", "walk1_subject1", *argv])
    args.play = evaluation  # playback loads weights only, so policy-only exports work
    env_cfg, train_cfg = task_registry.get_cfgs("g1_mimic", args)
    train_cfg.world_model.compile = train_cfg.imagination.compile = None
    if evaluation:
        configure_evaluation(env_cfg)
    env = task_registry.make_env("g1_mimic", env_cfg)
    return env, task_registry.make_alg_runner(env, "g1_mimic", args, train_cfg, log_root=None)


@needs_motions
def test_env_step_and_checkpoint_round_trip(tmp_path):
    env, runner = _make(["--num_envs", "4"])
    try:
        assert env.num_obs == 930 and env.num_privileged_obs == 689
        for _ in range(60):
            obs, rewards, _, _ = env.step(torch.zeros(4, 29))
        assert torch.isfinite(rewards).all() and env.replay.size > 0
        path = tmp_path / "model_0.pt"
        runner.save(path)
        action = runner.get_inference_policy("cpu")(obs)
        runner.policy.decoder.mlp.net[-1].bias.data += 1.0
        runner.load(path)
        torch.testing.assert_close(runner.get_inference_policy("cpu")(obs), action)
    finally:
        env.close()


@needs_motions
@pytest.mark.skipif(not CHECKPOINTS, reason="place a trained checkpoint in checkpoints/")
@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda p: p.name)
def test_checkpoint_tracks_walk(checkpoint):
    """A trained policy in checkpoints/ must track walk1_subject1 for 20 s."""
    env, runner = _make(["--num_envs", "1", "--checkpoint_path", str(checkpoint)], evaluation=True)
    try:
        policy, obs, total = runner.get_inference_policy("cpu"), env.get_observations(), 0.0
        with torch.no_grad():
            for _ in range(1000):
                obs, reward, done, _ = env.step(policy(obs))
                assert not done.any(), "policy fell"
                total += float(reward[0])
        assert total / (env.heads.reward_cfg.max_step_reward * 1000) > 0.85
    finally:
        env.close()
