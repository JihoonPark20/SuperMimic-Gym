"""Differentiable batched forward kinematics for the Unitree G1 (torch).

Built from the task's ``MjModel`` so the tree, joint axes, anchors and ``qpos0`` are
MuJoCo's. ``qpos`` (36) and ``qvel`` (35; free joint: world linear, body angular velocity)
give, per body, the link-origin pose and the world linear/angular link velocity
(MuJoCo ``cvel`` transported to the body origin), which are the quantities the
tracking reward and world-model heads consume.
"""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_gym import LEGGED_GYM_RESOURCES_DIR

from .rotations import quat_apply, quat_from_angle_axis, quat_mul, quat_normalize

G1_JOINT_NAMES = tuple(
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
)


@lru_cache(maxsize=1)
def load_g1_model():
    """Plain kinematic G1 model (no task actuators/sensors); joint order is the CSV/policy order."""
    model = mujoco.MjModel.from_xml_path(str(Path(LEGGED_GYM_RESOURCES_DIR) / "robots/g1/g1_mjlab.xml"))
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or names != G1_JOINT_NAMES:
        raise ValueError("G1 model joint order differs from the SuperMimic joint order")
    return model


class G1Kinematics:
    def __init__(self, model=None, device="cpu", dtype=torch.float32):
        model = model or load_g1_model()
        self.device, self.dtype = torch.device(device), dtype
        self.nbody = model.nbody - 1
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        self.parent, self.hinge = [], []
        jnt_pos, jnt_axis, qpos0 = np.zeros((self.nbody, 3)), np.zeros((self.nbody, 3)), np.zeros(self.nbody)
        for i, b in enumerate(range(1, model.nbody)):
            parent = int(model.body_parentid[b])
            self.parent.append(parent - 1)
            hinge = None
            if model.body_jntnum[b]:
                if model.body_jntnum[b] != 1:
                    raise ValueError("Only one joint per body is supported")
                j = int(model.body_jntadr[b])
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    hinge = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
                    jnt_pos[i], jnt_axis[i], qpos0[i] = model.jnt_pos[j], model.jnt_axis[j], model.qpos0[hinge[0]]
                elif i != 0:
                    raise ValueError("Only the root body may have a free joint")
            self.hinge.append(hinge)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)  # noqa: E731
        self.body_pos, self.body_quat = t(model.body_pos[1:]), t(model.body_quat[1:])
        self.jnt_pos, self.jnt_axis, self.qpos0 = t(jnt_pos), t(jnt_axis), t(qpos0)

    def body_index(self, name):
        return self.body_names.index(name)

    def body_indices(self, names):
        return torch.tensor([self.body_names.index(n) for n in names], device=self.device)

    def forward(self, qpos, qvel=None):
        batch = qpos.shape[:-1]
        nb = self.nbody
        pos, quat, lin, ang = [None] * nb, [None] * nb, [None] * nb, [None] * nb
        pos[0], quat[0] = qpos[..., 0:3], quat_normalize(qpos[..., 3:7])
        if qvel is not None:
            lin[0], ang[0] = qvel[..., 0:3], quat_apply(quat[0], qvel[..., 3:6])
        for i in range(1, nb):
            p = self.parent[i]
            b_pos = pos[p] + quat_apply(quat[p], self.body_pos[i].expand(*batch, 3))
            b_quat = quat_mul(quat[p], self.body_quat[i].expand(*batch, 4))
            hinge = self.hinge[i]
            if hinge is not None:
                jp, axis = self.jnt_pos[i].expand(*batch, 3), self.jnt_axis[i].expand(*batch, 3)
                anchor = b_pos + quat_apply(b_quat, jp)
                axis_w = quat_apply(b_quat, axis)
                b_quat = quat_mul(b_quat, quat_from_angle_axis(qpos[..., hinge[0]] - self.qpos0[i], axis))
                b_pos = anchor - quat_apply(b_quat, jp)
                if qvel is not None:
                    w_joint = axis_w * qvel[..., hinge[1], None]
                    ang[i] = ang[p] + w_joint
                    lin[i] = (
                        lin[p]
                        + torch.cross(ang[p], b_pos - pos[p], dim=-1)
                        + torch.cross(w_joint, b_pos - anchor, dim=-1)
                    )
            elif qvel is not None:
                ang[i] = ang[p]
                lin[i] = lin[p] + torch.cross(ang[p], b_pos - pos[p], dim=-1)
            pos[i], quat[i] = b_pos, b_quat
        out = {"pos": torch.stack(pos, dim=-2), "quat": torch.stack(quat, dim=-2)}
        if qvel is not None:
            out["lin_vel"], out["ang_vel"] = torch.stack(lin, dim=-2), torch.stack(ang, dim=-2)
        return out

    __call__ = forward
