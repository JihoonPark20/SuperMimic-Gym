from legged_gym.utils.task_registry import task_registry

from .g1_mimic.g1_mimic import G1Mimic
from .g1_mimic.g1_mimic_config import G1MimicCfg, G1MimicCfgSuperMimic

task_registry.register("g1_mimic", G1Mimic, G1MimicCfg(), G1MimicCfgSuperMimic())
