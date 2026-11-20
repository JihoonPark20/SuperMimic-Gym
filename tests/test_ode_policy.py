"""ODE policy semantics, task isolation, and end-to-end training/checkpoint compatibility."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.scripts.export_policy import main as export_policy
from legged_gym.supermimic.benchmark import run_benchmark, summarize
from legged_gym.supermimic.nets import ACT_DIM, Policy, PolicyCfg
from legged_gym.supermimic.world_model import OBS_DIM, flatten_history_term_major
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


def _policy(actor="ode", **kwargs):
    return Policy(PolicyCfg(actor=actor, encoder_hidden=(16,), decoder_hidden=(16,), **kwargs))


def test_previous_action_and_mlp_compatibility():
    torch.manual_seed(0)
    hist = torch.randn(10, 3, OBS_DIM)
    flat, goal = flatten_history_term_major(hist), torch.randn(3, 689)
    mlp, ode = _policy("mlp"), _policy()
    ode.load_state_dict(mlp.state_dict(), strict=True)
    torch.testing.assert_close(flat[:, -ACT_DIM:], hist[-1, :, -ACT_DIM:])
    torch.testing.assert_close(ode(flat, goal), hist[-1, :, -ACT_DIM:] + mlp(flat, goal))
    assert sum(p.numel() for p in mlp.parameters()) == sum(p.numel() for p in ode.parameters())


def test_ode_clamps_per_joint_and_keeps_time_gradient():
    bounds = tuple(0.1 + j / 100 for j in range(ACT_DIM))
    policy = _policy(ode_dmax=0.05, ode_amax=bounds)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.decoder.mlp.net[-1].bias.fill_(1.0)
    hist, goal = torch.zeros(10, 2, OBS_DIM), torch.zeros(2, 689)
    first = policy(flatten_history_term_major(hist), goal)
    torch.testing.assert_close(first, torch.full_like(first, 0.05))
    hist[-1, :, -ACT_DIM:] = 1.0
    torch.testing.assert_close(policy(flatten_history_term_major(hist), goal), torch.tensor(bounds).expand(2, -1))
    with torch.no_grad():
        policy.decoder.mlp.net[-1].bias.zero_()
    hist = torch.zeros(10, 2, OBS_DIM, requires_grad=True)
    initial = hist
    for _ in range(4):
        action = policy(flatten_history_term_major(hist), goal)
        obs = torch.cat((torch.zeros(2, OBS_DIM - ACT_DIM), action), dim=-1)
        hist = torch.cat((hist[1:], obs[None]))
    action.sum().backward()
    # Zero field leaves exactly the identity path across all four imagined actions.
    torch.testing.assert_close(initial.grad[-1, :, -ACT_DIM:], torch.ones(2, ACT_DIM))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actor": "other"},
        {"ode_dmax": 0},
        {"ode_dmax": float("nan")},
        {"ode_amax": ()},
        {"ode_amax": (1, 2)},
        {"ode_amax": (-1,)},
        {"ode_amax": (float("inf"),)},
    ],
)
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        PolicyCfg(**kwargs)


def test_task_presets_and_configuration_isolation():
    env, mlp = task_registry.get_cfgs("supermimic")
    ode_env, ode = task_registry.get_cfgs("supermimic-ode")
    assert class_to_dict(env) == class_to_dict(ode_env)
    assert mlp.policy.actor == "mlp" and ode.policy.actor == "ode"
    assert mlp.runner.experiment_name == "supermimic"
    assert ode.runner.experiment_name == "supermimic-ode"
    assert len(ode.policy.ode_amax) == ACT_DIM
    for section in ("world_model", "imagination", "supermimic", "critic"):
        assert class_to_dict(getattr(mlp, section)) == class_to_dict(getattr(ode, section))
    ode.policy.encoder_hidden[0] = 7
    assert task_registry.get_cfgs("supermimic-ode")[1].policy.encoder_hidden[0] == 2048
    assert mlp.policy.encoder_hidden[0] == 2048
    assert get_args(["--task", "supermimic-ode", "--rl_device", "cpu"]).task == "supermimic-ode"
    assert get_args(["--rl_device", "cpu"]).task == "g1_mimic"


@pytest.mark.parametrize("source,target", [("supermimic", "supermimic-ode"), ("supermimic-ode", "supermimic")])
def test_snapshot_cannot_override_requested_method(tmp_path, source, target):
    env, train = task_registry.get_cfgs(source)
    saved = {"task": source, "env": class_to_dict(env), "train": class_to_dict(train)}
    if source == "supermimic":
        del saved["train"]["policy"]["actor"]  # legacy snapshots are MLP
    (tmp_path / "run_config.json").write_text(json.dumps(saved))
    args = get_args(["--task", target, "--rl_device", "cpu", "--checkpoint_path", str(tmp_path / "model.pt")])
    with pytest.raises(ValueError, match="Checkpoint actor"):
        task_registry.get_cfgs(target, args)


def _small_runner(task, log_root=None, device="cpu", compile_mode=None):
    args = get_args(
        ["--task", task, "--rl_device", device, "--num_envs", "2", "--num_threads", "2", "--motion", "walk1_subject1"]
    )
    env_cfg, train = task_registry.get_cfgs(task, args)
    env_cfg.motion.replay_capacity = 64
    train.policy.encoder_hidden = train.policy.decoder_hidden = [32]
    train.world_model.hidden = [32]
    train.world_model.term_hidden = train.critic.hidden = [16]
    train.world_model.batch_size = train.imagination.batch_size = 4
    train.world_model.batches_per_update = 1
    train.world_model.window = 2
    train.imagination.horizon = 4
    train.imagination.critic_epochs = 1
    train.imagination.critic_minibatches = 2
    train.world_model.compile = train.imagination.compile = compile_mode
    train.supermimic.warmup_iterations = 4
    train.supermimic.updates_per_env_step = 0.5
    env = task_registry.make_env(task, env_cfg)
    return args, env, task_registry.make_alg_runner(env, task, args, train, log_root=log_root)


needs_motions = pytest.mark.skipif(
    not (Path(LEGGED_GYM_ROOT_DIR) / "data/motions/lafan1/walk1_subject1.npz").is_file(), reason="prepare motions"
)


@needs_motions
@pytest.mark.parametrize("task", ["supermimic", "supermimic-ode"])
def test_training_resume_export_and_method_guard(tmp_path, task):
    args, env, runner = _small_runner(task, tmp_path / "logs")
    try:
        before = [p.detach().clone() for p in runner.policy.parameters()]
        runner.learn(8)
        assert runner.num_updates == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, runner.policy.parameters()))
        metrics = runner.policy_trainer.update(runner.buffer)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert "dJda_first" in metrics and "imag_action_rate" in metrics
        if task.endswith("ode"):
            assert "ode_action_bound_fraction" in metrics
        obs = env.get_observations()
        expected = runner.get_inference_policy()(obs).detach()
        checkpoint = tmp_path / "model.pt"  # no sidecar: method guard must live in the checkpoint loader too
        runner.save(checkpoint)
        with torch.no_grad():
            runner.policy.decoder.mlp.net[-1].bias.add_(1)
        runner.load(checkpoint)
        torch.testing.assert_close(runner.get_inference_policy()(obs), expected)
        runner.learn(2)
        assert runner.current_learning_iteration == 10

        exported = tmp_path / "policy.pt"
        export_policy([str(checkpoint), str(exported)])
        runner.load(exported, restore_env=False)
        torch.testing.assert_close(runner.get_inference_policy()(obs), expected)
        with pytest.raises(ValueError, match="only the policy"):
            runner.load(exported, restore_env=True)

        ck = torch.load(exported, weights_only=False)
        ck["config"]["policy"]["actor"] = "mlp" if task.endswith("ode") else "ode"
        torch.save(ck, tmp_path / "wrong.pt")
        with pytest.raises(ValueError, match="Checkpoint actor"):
            runner.load(tmp_path / "wrong.pt", restore_env=False)

        ck["config"]["policy"].pop("actor")
        torch.save(ck, tmp_path / "legacy.pt")
        if task.endswith("ode"):
            with pytest.raises(ValueError, match="Checkpoint actor"):
                runner.load(tmp_path / "legacy.pt", restore_env=False)
            ck["config"]["policy"]["actor"] = "ode"
            ck["config"]["policy"]["ode_amax"] = None
            torch.save(ck, tmp_path / "wrong_bounds.pt")
            with pytest.raises(ValueError, match="configuration differs"):
                runner.load(tmp_path / "wrong_bounds.pt", restore_env=False)
            ck["config"]["policy"].pop("ode_amax")
            torch.save(ck, tmp_path / "incomplete.pt")
            with pytest.raises(ValueError, match="must include"):
                runner.load(tmp_path / "incomplete.pt", restore_env=False)
        else:
            runner.load(tmp_path / "legacy.pt", restore_env=False)

        # Saved run settings reconstruct the architecture for a new process.
        args.checkpoint_path = str(runner.log_dir / "model_8.pt")
        restored_env, restored_train = task_registry.get_cfgs(task, args)
        restored = task_registry.make_alg_runner(env, task, args, restored_train, log_root=None)
        assert restored.current_learning_iteration == 8 and restored.num_updates == 2
        assert restored.policy_cfg.actor == runner.policy_cfg.actor

        # Sensor/execution noise must never become the ODE's integration state.
        env.cfg.motion.action_noise_std = 0.5
        action = torch.full((2, ACT_DIM), 0.1)
        obs, _, done, _ = env.step(action)
        assert not torch.equal(env.executed_actions, action)
        torch.testing.assert_close(obs["policy"][~done, -ACT_DIM:], action[~done])
        previous_other = env.actions[1].clone()
        env.reset_idx(torch.tensor([0]))
        state = env.state()
        bodies = env.heads.robot_bodies(state)
        env.compute_observations(
            state, torch.cat((bodies["pos"][:, env.anchor_index], bodies["quat"][:, env.anchor_index]), -1)
        )
        torch.testing.assert_close(env.get_observations()["policy"][0, -ACT_DIM:], torch.zeros(ACT_DIM))
        torch.testing.assert_close(env.get_observations()["policy"][1, -ACT_DIM:], previous_other)

        captured = {}
        reset_idx = env.reset_idx

        def capture_reset(ids, finalize=True):
            captured["velocity"] = env.state()[:, 42:71].clone()
            reset_idx(ids, finalize=finalize)

        env.reset_idx = capture_reset
        env.max_episode_length = 1
        obs, _, done, extras = env.step(action)
        assert done.all()
        torch.testing.assert_close(extras["joint_vel"], captured["velocity"])
        torch.testing.assert_close(obs["policy"][:, -ACT_DIM:], torch.zeros(2, ACT_DIM))
    finally:
        env.close()


class _BenchmarkEnv:
    """Two clips on one auto-resetting env; an enormous reset jump must not affect smoothness."""

    num_envs = num_actions = 1
    dt = 0.5

    def __init__(self):
        self.motion = SimpleNamespace(
            num_clips=2,
            clip_start=torch.tensor([0, 10]),
            clip_len=torch.tensor([4, 8]),
            clip_names=["complete", "failed"],
        )
        self.heads = SimpleNamespace(reward_cfg=SimpleNamespace(max_step_reward=1.0))
        self.forced_frame = torch.zeros(1, dtype=torch.long)
        self.terminated_buf = torch.zeros(1, dtype=torch.bool)

    def reset(self):
        self.t = self.clip = 0
        return {}

    def step(self, action):
        self.t += 1
        self.executed_actions = torch.tensor([[float(self.t + 1000 * self.clip)]])
        # Constant joint acceleration; no joint jerk within either clip.
        velocity = 2 * self.executed_actions
        done = self.t == 4
        self.terminated_buf[:] = done and self.clip == 1
        self.term_rewards = {"total": torch.ones(1)}
        if done:
            self.t = 0
            self.clip += 1
        return {}, torch.ones(1), torch.tensor([done]), {"joint_vel": velocity}


def test_benchmark_full_length_and_smoothness_ignore_reset_boundaries():
    results = run_benchmark(_BenchmarkEnv(), lambda obs: None, verbose=False)
    assert [r["full_length_return"] for r in results] == [1.0, 0.5]
    for result in results:
        assert result["smooth"] == {"action_rate": 1.0, "action_acc": 0.0, "joint_acc": 4.0, "joint_jerk": 0.0}
        assert result["smooth_samples"]["action_rate"] == 3
        assert result["smooth_samples"]["joint_jerk"] == 2
    summary = summarize(results)
    assert summary["completion_rate"] == 0.5 and summary["mean_full_length_return"] == 0.75
    assert summary["smooth"]["joint_jerk"] == 0.0
