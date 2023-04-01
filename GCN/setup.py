from setuptools import setup, find_packages

setup(
    name='genco-tsp',
    version='1.0.0',
    author='Johanna Sommer, Simon Geisler',
    description='Official implementation of the paper "Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness"',
    url='',
    packages=['tsp/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)