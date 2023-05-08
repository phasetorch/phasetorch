from setuptools import setup, find_packages

with open("VERSION", "r") as fh:
    version = fh.read().strip()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name='phasetorch',
      version=version,
      description='PhaseTorch is a Python package for phase retrieval in X-ray phase contrast computed tomography (XPCT).',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url='https://github.com/phasetorch/phasetorch',
      license='GPL-2.0',
      packages=find_packages(where="./"),
      package_dir={"": "./"},
      package_data={"phasetorch": ["./data/*.h5"]},
      setup_requires=[],
      install_requires=[],
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ])
