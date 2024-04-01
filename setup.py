import setuptools

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="causal4",
    author="Kai Chen",
    author_email="kchen513@sjtu.edu.cn",

    version="0.0.1",
    url="https://github.com/NeoNeuron/causal4",

    description="Software package for studies of 4 causality measures.",

    install_requires=requirements,
    packages=setuptools.find_packages(),
)