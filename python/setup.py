#!/usr/bin/env python3
"""ProSeCo provides diverse functionality to evaluate, optimize and analyze the output of ProSeCo Planning."""

DOCLINES = (__doc__ or "").split("\n")
import setuptools

with open("./proseco/requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="proseco",
    version="0.7.0",
    install_requires=requirements,
    extras_require={"test": ["shapely"]},
    author="Karl Kurzer et al.",
    author_email="kurzer@kit.edu",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://git.scc.kit.edu/atks/dfg/ros_proseco_planning",
    packages=setuptools.find_packages(exclude=("tests")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    python_requires=">=3.8.5",
)
