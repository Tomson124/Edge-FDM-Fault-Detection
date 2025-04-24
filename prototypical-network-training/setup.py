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
