"""
/setup.py

Author: Jared Moore
Date: October, 2025
"""

import setuptools


def _read_requirements(path: str) -> list[str]:
    """Return a list of requirements from a requirements file."""

    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.read().splitlines()]
    return [line for line in lines if line and not line.startswith("-r")]


base_requirements = _read_requirements("requirements/base.txt")
viewer_requirements = _read_requirements("requirements/viewer.txt")
chatlog_requirements = _read_requirements("requirements/chatlog_processing_pipeline.txt")
full_requirements = _read_requirements("requirements/full.txt")

analysis_packages = ["analysis"] + [
    f"analysis.{pkg}" for pkg in setuptools.find_packages(where="analysis")
]
src_packages = setuptools.find_packages(where="src")

setuptools.setup(
    name="llm-delusions",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="Transcript Helpers",
    package_dir={"": "src", "analysis": "analysis"},
    packages=src_packages + analysis_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=base_requirements,
    extras_require={
        "viewer": viewer_requirements,
        "chatlog_processing_pipeline": chatlog_requirements,
        "full": full_requirements,
    },
    entry_points={
        "console_scripts": [
            "process_chats = chatlog_processing_pipeline.commands:main",
        ]
    },
)
