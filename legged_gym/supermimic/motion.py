"""Motion data: G1 retarget CSV -> 50 Hz NPZ conversion and the multi-clip motion library.

NPZ keys follow mjlab's motion format (fps, joint_pos/vel, body_{pos,quat,lin_vel,ang_vel}_w,
body_names), so files produced by supermimic-m5 / mjlab load unchanged.

Library layout: all clips concatenated on one global frame axis. Per frame it keeps the 71-d
reference state ``[root_pos 3, root_quat 4, joint_pos 29, root_lin_vel_w 3, root_ang_vel_w 3,
joint_vel 29]``, the tracked-body link states and the 68-d goal features
``[joint_pos - q0, joint_vel, ref projected gravity, ref root ang vel (root frame),
ref root lin vel (heading frame), ref root height]``.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from .kinematics import G1_JOINT_NAMES, load_g1_model
from .rotations import projected_gravity, quat_apply_inverse, yaw_quat

GOAL_FEAT_DIM = 68

# ----------------------------------------------------------------------------- CSV -> NPZ


def _np_quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _np_rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    return q[..., 1:] * np.where(s > 1e-9, angle / np.where(s > 1e-9, s, 1.0), 2.0)[..., None]


def _np_rotate_inverse(q, v):
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - q[..., :1] * t + np.cross(xyz, t)


def resample_csv(raw, input_fps, output_fps=50.0):
    """LAFAN1 retarget rows (pos m, quat xyzw, 29 joints rad) -> resampled motion with mjlab velocities."""
    if raw.ndim != 2 or raw.shape[1] != 7 + len(G1_JOINT_NAMES) or len(raw) < 2:
        raise ValueError(f"Expected (N>=2, 36) motion rows, got {raw.shape}")
    out_dt = 1.0 / output_fps
    duration = (len(raw) - 1) / input_fps
    n_out = math.floor(duration / out_dt + 1e-6) + 1
    phase = np.arange(n_out) * out_dt / duration * (len(raw) - 1)
    i0 = np.clip(np.floor(phase).astype(int), 0, len(raw) - 1)
    i1 = np.minimum(i0 + 1, len(raw) - 1)
    blend = (phase - i0)[:, None]
    quat = raw[:, [6, 3, 4, 5]]
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    root_pos = raw[i0, :3] * (1 - blend) + raw[i1, :3] * blend
    joint_pos = raw[i0, 7:] * (1 - blend) + raw[i1, 7:] * blend
    q0, q1 = quat[i0], quat[i1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1, dot = np.where(dot < 0, -q1, q1), np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    linear = dot > 0.9995
    sin_theta = np.where(linear, 1.0, np.sin(theta))
    w0 = np.where(linear, 1 - blend, np.sin((1 - blend) * theta) / sin_theta)
    w1 = np.where(linear, blend, np.sin(blend * theta) / sin_theta)
    root_quat = w0 * q0 + w1 * q1
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    root_ang = np.zeros((n_out, 3))
    if n_out >= 3:
        omega = _np_rotvec(_np_quat_mul(root_quat[2:], root_quat[:-2] * [1, -1, -1, -1])) / (2 * out_dt)
        root_ang = np.concatenate((omega[:1], omega, omega[-1:]))
    return {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_pos": joint_pos,
        "root_lin_vel_w": np.gradient(root_pos, out_dt, axis=0),
        "root_ang_vel_w": root_ang,
        "joint_vel": np.gradient(joint_pos, out_dt, axis=0),
    }


def convert_csv_to_npz(csv_path, out_path, input_fps=30.0, output_fps=50.0, model=None):
    model = model or load_g1_model()
    m = resample_csv(np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2), input_fps, output_fps)
    data = mujoco.MjData(model)
    T, nb = len(m["root_pos"]), model.nbody - 1
    pos, quat, lin, ang = (np.empty((T, nb, d)) for d in (3, 4, 3, 3))
    ang_b = _np_rotate_inverse(m["root_quat"], m["root_ang_vel_w"])
    root = model.jnt_bodyid[0]
    for t in range(T):
        data.qpos[:] = np.concatenate((m["root_pos"][t], m["root_quat"][t], m["joint_pos"][t]))
        data.qvel[:] = np.concatenate((m["root_lin_vel_w"][t], ang_b[t], m["joint_vel"][t]))
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        cvel = data.cvel[1:]
        pos[t], quat[t], ang[t] = data.xpos[1:], data.xquat[1:], cvel[:, :3]
        lin[t] = cvel[:, 3:] - np.cross(cvel[:, :3], data.subtree_com[root][None] - data.xpos[1:])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        fps=np.array([output_fps], dtype=np.float32),
        joint_pos=m["joint_pos"].astype(np.float32),
        joint_vel=m["joint_vel"].astype(np.float32),
        body_pos_w=pos.astype(np.float32),
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=lin.astype(np.float32),
        body_ang_vel_w=ang.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array([model.body(i).name for i in range(1, model.nbody)]),
        input_fps=np.array([input_fps], dtype=np.float32),
    )
    temporary.replace(out_path)
    return T


def discover_npz(directory, names=None, limit=None):
    files = sorted(Path(directory).glob("*.npz"))
    if names:
        missing = set(names) - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(f"Motions not found under {directory}: {sorted(missing)}")
        files = [f for f in files if f.stem in set(names)]
    return files[:limit] if limit else files


# ----------------------------------------------------------------------------- library


@dataclass
class RefState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_vel: torch.Tensor


@dataclass
class RefBodies:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


def goal_features(ref, default_joint_pos):
    q = ref.root_quat_w
    return torch.cat(
        (
            ref.joint_pos - default_joint_pos,
            ref.joint_vel,
            projected_gravity(q),
            quat_apply_inverse(q, ref.root_ang_vel_w),
            quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w),
            ref.root_pos_w[..., 2:3],
        ),
        dim=-1,
    )


class MotionLibrary:
    def __init__(self, npz_files, body_names, default_joint_pos, device="cpu", bin_frames=50):
        if not npz_files:
            raise ValueError("MotionLibrary needs at least one clip")
        self.device = torch.device(device)
        self.body_names = tuple(body_names)
        self.bin_frames = int(bin_frames)
        self.default_joint_pos = torch.as_tensor(np.asarray(default_joint_pos), dtype=torch.float32, device=self.device)
        refs, bodies, self.clip_names, lengths = [], {k: [] for k in ("pos", "quat", "lin", "ang")}, [], []
        for path in npz_files:
            with np.load(path) as d:
                if float(d["fps"][0]) != 50.0 or d["joint_pos"].shape[1] != len(G1_JOINT_NAMES):
                    raise ValueError(f"{path}: expected 50 Hz G1 motion")
                names = [str(n) for n in d["body_names"]]
                idx, root = [names.index(n) for n in self.body_names], names.index("pelvis")
                bp, bq, bl, ba = d["body_pos_w"], d["body_quat_w"], d["body_lin_vel_w"], d["body_ang_vel_w"]
                refs.append(
                    np.concatenate(
                        (bp[:, root], bq[:, root], d["joint_pos"], bl[:, root], ba[:, root], d["joint_vel"]), 1
                    )
                )
                for key, value in zip(bodies, (bp, bq, bl, ba)):
                    bodies[key].append(value[:, idx])
            self.clip_names.append(Path(path).stem)
            lengths.append(len(refs[-1]))
        t = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float32, device=self.device)  # noqa: E731
        self._ref = t(refs)
        self.body_pos_w, self.body_quat_w = t(bodies["pos"]), t(bodies["quat"])
        self.body_lin_vel_w, self.body_ang_vel_w = t(bodies["lin"]), t(bodies["ang"])
        self.goal_feat = goal_features(self._split(self._ref), self.default_joint_pos)
        self.num_clips, self.num_frames = len(lengths), int(self._ref.shape[0])
        self.clip_len = torch.tensor(lengths, device=self.device)
        self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)
        bins = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins, 0) - bins
        self.num_bins = int(bins.sum())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), bins)
        local = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local * self.bin_frames
        self.bin_end = torch.minimum(
            self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip]
        )

    @staticmethod
    def _split(r):
        return RefState(r[..., 0:3], r[..., 3:7], r[..., 7:36], r[..., 36:39], r[..., 39:42], r[..., 42:71])

    def ref(self, frame):
        return self._split(self._ref[frame])

    def bodies(self, frame):
        return RefBodies(
            self.body_pos_w[frame], self.body_quat_w[frame], self.body_lin_vel_w[frame], self.body_ang_vel_w[frame]
        )

    def bin_of_frame(self, frame):
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame):
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame, k):
        idx = torch.minimum(frame[:, None] + torch.arange(k, device=self.device), self.last_frame_of(frame)[:, None])
        return self.goal_feat[idx].reshape(frame.shape[0], k * GOAL_FEAT_DIM)

    def summary(self):
        return f"MotionLibrary: {self.num_clips} clips, {self.num_frames} frames ({self.num_frames / 50 / 3600:.3f} h), {self.num_bins} bins"
