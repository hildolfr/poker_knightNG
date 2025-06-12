#!/usr/bin/env python3
"""Setup script for poker-knight-ng."""

from setuptools import setup, find_packages
import os

def read_long_description():
    """Read the contents of README.md if it exists."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "GPU-accelerated Texas Hold'em poker solver"

setup(
    name="poker-knight-ng",
    version="0.1.0",
    author="Hildolfr",
    description="GPU-accelerated Texas Hold'em poker solver",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/hildolfr/poker_knightNG",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "poker_knight_ng": ["cuda/*.cu", "cuda/*.cuh"],
    },
    python_requires=">=3.8",
    install_requires=[
        "cupy-cuda12x>=13.0",
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "mypy",
            "pytest-benchmark",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: CUDA",
    ],
    keywords="poker, texas-holdem, gpu, cuda, monte-carlo, solver",
)