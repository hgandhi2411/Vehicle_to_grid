from setuptools import setup

with open('README.md') as f:
    long_description = ''.join(f.readlines())

setup(name='v2g',
      version='0.01',
      description='Vehicle-To-Grid Simulation',
      long_description=long_description,
      author='Heta Gandhi',
      packages=['v2g'],
      install_requires=[
          'matplotlib',
          'seaborn',
          'numpy',
          'pandas',
          'scipy'],
      scripts=['bin/v2g-optimize']
     )
