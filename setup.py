from setuptools import find_packages, setup

setup(
    name='watch_price_predictor',
    version='0.0.1', # v2 (MLFlow) + v3 (API)
    author='(Enes Guler) - enesml',
    author_email='(enesguler.ml@gmail.com)',
    packages=find_packages(),
    install_requires=[],
    description='End-to-end MLOps pipeline for watch price regression (v3 API).'
)