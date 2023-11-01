import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(
    name = "automix-llm",
    version = "1.0.3",
    description = ("Library for Mixing Multiple Models with differential costs and performance."),
    license = "Apache License 2.0",
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tqdm'
    ]
)