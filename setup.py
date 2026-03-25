import os
import re

from setuptools import setup, find_packages

cur_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def find_version(*file_paths):
    with open(os.path.join(cur_dir, *file_paths)) as fp:
        match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="MARSim",
    author="Shae McFadden",
    license="MIT",
    version=find_version("MARSim", "__init__.py"),
    description="Multi-Agent Resupply Simulator for contested and partially observable battlefield scenarios.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McFadden-S/MARSim",
    install_requires=[
        "gymnasium==0.28.1",
        "numpy>=1.19.2",
        "torch>=1.13.0",
        "pygame>=2.1.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.5"],
        "plots": ["matplotlib>=3.5.0", "Pillow>=9.0.0"],
    },
    package_dir={"": "./"},
    packages=find_packages(where="./", include="MARSim*"),
    include_package_data=True,
    python_requires=">=3.9",
)
