from setuptools import setup, find_packages

setup(name='phasetorch',
      version='0.1.0',
      description='Algorithms for phase constrast tomography reconstruction',
      url='',
      author='K. Aditya Mohan',
      author_email='mohan3@llnl.gov',
      license='MIT',
      packages=find_packages(where="./"),
      package_dir={"": "./"},
      package_data={"phasetorch": ["./data/*.h5"]},
      setup_requires=[],
      install_requires=[],
      zip_safe=False)
