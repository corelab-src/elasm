from setuptools import setup, find_packages

setup(
    name='hecate',
    version='0.0.1',
    description='PYPI package for hecate binding',
    author='yongwoo',
    author_email='dragonrain96@gmail.com',
    install_requires=['numpy'],
    packages=find_packages(exclude=[]),
    keywords=['yongwoo', 'homomorphic encryption', 'ckks', 'hecate', 'elasm'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
