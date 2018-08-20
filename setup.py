from setuptools import setup

setup(
    name='gerrymetrics',
    version='1.0dev',
    packages=setuptools.find_packages(),
    description='Lots of metrics for quantifying gerrymandering',
    author='Princeton Gerrymandering Project',
    author_email='wtadler@princeton.edu',
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    install_requires=['pandas', 'numpy', 'scipy', 'matplotlib', 'tqdm']
)