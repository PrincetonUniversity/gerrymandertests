import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='gerrymetrics',
    version='0.1.3',
    packages=setuptools.find_packages(),
    description='Lots of metrics for quantifying gerrymandering',
    author='Princeton Gerrymandering Project',
    author_email='wtadler@princeton.edu',
    license='GNU General Public License v3.0',
    long_description=long_description,
    install_requires=['pandas', 'numpy', 'scipy', 'matplotlib', 'tqdm'],
    url='https://github.com/PrincetonUniversity/gerrymandertests'
)
