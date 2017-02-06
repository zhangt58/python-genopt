from setuptools import setup, find_packages
import versioneer
import os

def readme():
    with open('README.rst') as f:
        return f.read()

requiredpackages = ['numpy', 'matplotlib']#, 'flame']

setup(
        name = "genopt",
        version = versioneer.get_version(),
        cmdclass = versioneer.get_cmdclass(),
        description = "General multi-dimensional optimization package",
        long_description = readme() + '\n\n',
        author = "Tong Zhang",
        author_email = "zhangt@frib.msu.edu",
        platforms = ["Linux"],
        license = "MIT",
        url = "http://archman.github.io/genopt/",
        packages = find_packages(),
        requires = requiredpackages,
)
