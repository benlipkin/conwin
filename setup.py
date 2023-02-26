from setuptools import find_packages, setup

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    name="conwin",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
