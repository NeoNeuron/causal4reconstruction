import setuptools
import sys
import numpy
import sysconfig

setuptools.setup(
    name="causal4",
    author="Kai Chen",
    author_email="kchen513@sjtu.edu.cn",

    version="0.0.1",
    url="https://github.com/NeoNeuron/causal4",

    description="Software package for studies of 4 causality measures.",

    install_requires=['numpy', 'matplotlib', 'scipy', 'networkx'],
    packages=setuptools.find_packages(),
)