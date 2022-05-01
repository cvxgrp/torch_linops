#!/usr/bin/python
from setuptools import setup

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
    name="torch-linops",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "torch",
    ],
    url="https://github.com/cvxgrp/torch_linops",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Parth Nobel",
    author_email="ptnobel@stanford.edu",
)
