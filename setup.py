from setuptools import find_packages, setup

setup(
    name="supermimic-gym",
    version="0.1.0",
    license="BSD-3-Clause",
    license_files=["LICENSE", "licenses/*", "resources/robots/UNITREE_LICENSE"],
    description="SuperMimic whole-body motion tracking for the Unitree G1 in a legged_gym interface",
    python_requires=">=3.10,<3.14",
    packages=find_packages(include=["legged_gym", "legged_gym.*"]) + ["legged_gym.resources"],
    package_dir={"legged_gym.resources": "resources"},
    package_data={
        "legged_gym.resources": [
            "robots/UNITREE_LICENSE",
            "robots/g1/g1_mjlab.xml",
            "robots/g1/unitree_mjlab.json",
            "robots/g1/meshes/*",
            "motions/lafan1_manifest.json",
        ]
    },
    install_requires=["mujoco==3.17.0", "torch>=2.6.0", "numpy>=1.26", "tensorboard"],
    extras_require={"dev": ["pytest", "ruff"]},
    entry_points={
        "console_scripts": ["legged-train=legged_gym.scripts.train:main", "legged-play=legged_gym.scripts.play:main"]
    },
)
