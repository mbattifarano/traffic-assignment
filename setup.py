import os
from setuptools import setup, find_packages
from Cython.Build import cythonize


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
    ext_modules=cythonize("src/traffic_assignment/*.pyx",
                          language_level=3),
    zip_safe=False,
    test_requires=read_requirements('test.txt'),
    install_requires=read_requirements('install.txt'),
    python_requires=">=3.7",
)
