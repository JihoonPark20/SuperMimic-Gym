"""Batched torch quaternion helpers (wxyz), differentiable, arbitrary leading dims.

Same conventions as mjlab/IsaacLab ``lab_api.math``, which the ported SuperMimic code assumes;
adds the rotation-vector maps, 6D rotations and heading frames used by the world model.
"""

import math

import torch

_EPS = 1.0e-8


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q):
    return quat_conjugate(q) / (q * q).sum(-1, keepdim=True).clamp_min(1e-9)


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q):
    """Flip sign so that w >= 0."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_apply(q, v):
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q, v):
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_from_angle_axis(angle, axis):
    half = 0.5 * angle.unsqueeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def axis_angle_from_quat(q, eps=1.0e-6):
    q = quat_canonical(q)
    mag = q[..., 1:].norm(dim=-1)
    angle = 2.0 * torch.atan2(mag, q[..., 0])
    k = torch.where(angle.abs() > eps, torch.sin(0.5 * angle) / angle, 0.5 - angle * angle / 48.0)
    return q[..., 1:] / k.unsqueeze(-1)


def quat_box_minus(q1, q2):
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_error_magnitude(q1, q2):
    return quat_box_minus(q1, q2).norm(dim=-1)


def quat_from_rotvec(v):
    """Exponential map with a series fallback so gradients stay finite at v = 0."""
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    small = angle < 1e-4
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(0.5 * angle) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(0.5 * angle))
    return torch.cat((w, v * k), dim=-1)


def rotvec_from_quat(q):
    q = quat_canonical(q)
    xyz = q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., :1])
    small = s < 1e-6
    k = torch.where(small, 2.0 + angle * angle / 12.0, angle / torch.where(small, torch.ones_like(s), s))
    return xyz * k


def rot6d_from_quat(q):
    return matrix_from_quat(q)[..., :, :2].reshape(q.shape[:-1] + (6,))


def heading_yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=-1)


def yaw_quat(q):
    return quat_from_yaw(heading_yaw(q))


heading_quat = yaw_quat


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def projected_gravity(q):
    g = torch.tensor([0.0, 0.0, -1.0], dtype=q.dtype, device=q.device).expand(q.shape[:-1] + (3,))
    return quat_apply_inverse(q, g)
