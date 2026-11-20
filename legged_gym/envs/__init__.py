from legged_gym.utils.task_registry import task_registry

from .g1_mimic.g1_mimic import G1Mimic
from .g1_mimic.g1_mimic_config import G1MimicCfg, G1MimicCfgSuperMimic, G1MimicCfgSuperMimicODE

for name, config in (
    ("g1_mimic", G1MimicCfgSuperMimic),  # legacy task and log directory
    ("supermimic", G1MimicCfgSuperMimic),
    ("supermimic-ode", G1MimicCfgSuperMimicODE),
):
    train_cfg = config()
    train_cfg.runner.experiment_name = name
    task_registry.register(name, G1Mimic, G1MimicCfg(), train_cfg)
