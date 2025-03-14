from setuptools import setup, find_packages

setup(
    name="ldpc_neural_decoder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
    ],
) 