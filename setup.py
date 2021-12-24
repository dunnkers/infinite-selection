#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name='infinite_selection',
      version='0.1.0',
      description='Infinite Feature Selection (Roffo et al) distributed as a Python 3 package.',
      author='Giorgo Roffo, Simone Melzi, Marco Cristani',
      packages=find_packages(),
      install_requires=[
            "numpy>=1.19",
            "scikit-learn>=0.24",
      ],
      )
