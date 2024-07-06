from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """

    :param file_path: file path to download the required packages
    :return: gives out list of packages
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='Phishing URL Detection',
    version='0.0.1',
    author='Himala Praharsha Chittathuru',
    author_email='chimalapraharsha21@iisertvm.ac.in',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
