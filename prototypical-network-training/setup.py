"""
This code was adapted from the original implementation of Prototypical Networks for Few-Shot Learning.
Link: https://github.com/jakesnell/prototypical-networks
"""

from setuptools import setup, find_packages

setup(name='protonets',
      version='0.0.2',
      author='Jake Snell',
      author_email='jsnell@cs.toronto.edu',
      maintainer='Oliver Bravery',
      maintainer_email='dev@oliverbravery.uk',
      license='MIT',
      packages=find_packages(where='.', include=['protonets', 'protonets.*']),
      install_requires=[
          'torch',
          'torchvision',
          'tqdm',
          'ptflops'
      ])
