#Setup.py helps package the project which can be later used to upload on the websites such as Pypi.org etc.

from setuptools import setup, find_packages   # type: ignore
from typing import List

HYPHEN_E = "-e ."

def get_requirements(filepath: str) -> List[str]:
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]

        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)


setup(name='ML_Pipeline_Project',
    version='1.0',
    description='A Machine Learning Pipeline Project',
    author='Syed Haadi Raza',
    author_email='Karrar.haider128@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)