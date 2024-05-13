# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "scipy",
    "hydra-core",
    "yacs",
    "h5py",
    "pybullet",
    "pygifsicle",
    "open3d",
    "numpy-quaternion==2022.4.3",
    "pybind11-global",
    "sophuspy",
    # "trimesh",
    "pin>=2.6.17",
    # "torch_cluster",
    # "torch_scatter",
    # "pillow==9.5.0",  # For Detic compatibility
    "pyqt6"
]

setup(
    name="home-robot",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
