"""Batched CPU MuJoCo for the flat G1 (mjlab 1.2 model construction, native mujoco.batch).

Collision, actuator and solver settings follow mjlab's G1 so policies transfer between the two.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch

from legged_gym import LEGGED_GYM_RESOURCES_DIR


def joint_parameters(names, patterns, default=None):
    """Resolve regex settings in insertion order; later (more specific) entries win."""
    values = np.full(len(names), np.nan if default is None else default, dtype=np.float64)
    for pattern, value in patterns.items():
        mask = [re.fullmatch(pattern, name) is not None for name in names]
        if not any(mask):
            raise ValueError(f"Joint pattern matches no joints: {pattern}")
        values[mask] = value
    if not np.isfinite(values).all():
        raise ValueError("Every joint must have a finite configured value")
    return values


def load_model(cfg):
    path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(path.parent / compiler.get("meshdir", "")))
    s = cfg.sim
    ET.SubElement(
        root,
        "option",
        timestep=str(s.dt),
        integrator="implicitfast",
        cone=s.cone,
        solver="Newton",
        jacobian="auto",
        impratio="1",
        iterations=str(s.iterations),
        ls_iterations=str(s.ls_iterations),
        tolerance=str(s.tolerance),
        ls_tolerance=str(s.ls_tolerance),
        ccd_iterations=str(s.ccd_iterations),
        gravity="0 0 -9.81",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="1 0.005 0.0001",
        rgba="0.25 0.3 0.35 1",
        condim="3",
    )
    ET.SubElement(world, "light", name="task_light", pos="0 0 4", dir="0 0 -1", directional="true")
    {b.get("name"): b for b in world.iter("body")}[cfg.asset.root_body].set(
        "pos", " ".join(map(str, cfg.init_state.pos))
    )
    feet = set(cfg.asset.foot_geoms)
    for geom in world.iter("geom"):
        name = geom.get("name", "")
        if name.endswith("_collision"):
            # Self-collisions on; feet get torsional friction and win contact priority (mjlab FULL_COLLISION).
            geom.attrib.update(
                contype="1", conaffinity="1", condim="3" if name in feet else "1", priority="1" if name in feet else "0"
            )
            if name in feet:
                geom.set("friction", "0.6 0.005 0.0001")
    c, names = cfg.control, cfg.asset.joint_names
    actuators = ET.SubElement(root, "actuator")
    for name, arm, kp, kd, limit in zip(
        names, *(joint_parameters(names, p) for p in (c.armature, c.stiffness, c.damping, c.effort_limits))
    ):
        world.find(f".//joint[@name='{name}']").attrib.update(armature=str(arm), frictionloss="0")
        ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=str(kp),
            kv=str(kd),
            forcelimited="true",
            forcerange=f"{-limit} {limit}",
            ctrllimited="false",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class MujocoBatch:
    def __init__(self, cfg, rng):
        path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
        digest = hashlib.sha256(path.read_bytes())
        meshdir = path.parent / ET.parse(path).getroot().find("compiler").get("meshdir", "")
        for mesh in ET.parse(path).getroot().findall("asset/mesh"):
            digest.update((meshdir / mesh.get("file")).read_bytes())
        self.asset_digest = digest.hexdigest()
        self.model = load_model(cfg)
        self.batch = batch.Batch(self.model, nsim=cfg.env.num_envs, nthread=cfg.sim.nthread)
        self.qpos, self.qvel, self.ctrl = (self.batch.bind(k) for k in ("qpos", "qvel", "ctrl"))
        self.joint_ids = np.array([self.model.joint(n).id for n in cfg.asset.joint_names])
        self.encoder_bias = np.zeros((cfg.env.num_envs, len(cfg.asset.joint_names)))
        dr = cfg.domain_rand
        if dr.randomize_com:  # startup only: COM constants feed mj_setConst
            com = self.batch.expand("body_ipos")
            com[:, self.model.body(cfg.asset.mass_body).id] += rng.uniform(*dr.com_range, (cfg.env.num_envs, 3))
            self.batch.set_const()
        self.batch.reset()

    def step(self, targets):
        self.ctrl[:] = targets
        self.batch.step(stage=mujoco.mjtStage.mjSTAGE_NONE)
        warnings = [
            int(w)
            for w in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
        ]
        failed = np.any(self.batch.warning[:, warnings, 1] > 0, axis=1)
        if failed.any():
            raise FloatingPointError(
                f"MuJoCo reported invalid dynamics in environments {np.flatnonzero(failed).tolist()}"
            )

    def reset(self, ids):
        self.batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)

    def forward(self, ids=None):
        self.batch.forward(ids)

    def close(self):
        self.qpos = self.qvel = self.ctrl = self.batch = None
