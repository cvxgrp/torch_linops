from setuptools import setup

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
    name="torch-linops",
    version="0.1.0",
    packages=["linops"],
    license="GPLv3",
    description="A library to define abstract linear operators, and associated algebra and matrix-free algorithms, that works with pyTorch Tensors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "torch",
        "scipy"
    ],
    url="https://github.com/cvxgrp/torch_linops",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Parth Nobel",
    author_email="ptnobel@stanford.edu",
)
