from setuptools import setup, find_packages

import os

try:
    long_description = open("../../README.md").read()
except FileNotFoundError:
    long_description = "Python SDK for T-VectorDB"

setup(
    name="tvectordb",
    version="0.1.0",
    description="Python SDK for T-VectorDB — High-Performance gRPC Vector Database Client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kavin Karthick",
    url="https://github.com/kavinkarthicknattuthurai/T-VectorDB",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.50.0",
        "protobuf>=4.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
