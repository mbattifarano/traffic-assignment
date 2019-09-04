import os
from setuptools import setup, find_packages


def read_requirements(fname):
    return open(os.path.join('requirements', fname)).read().strip().splitlines()


setup(
    name="traffic-assignment",
    version="0.1",
    description="Traffic assignment solver for python.",
    author="Matt Battifarano",
    author_email="mbattifa@andrew.cmu.edu",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    test_requires=read_requirements('test.txt'),
    install_requires=read_requirements('install.txt'),
)
