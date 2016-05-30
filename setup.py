#!/usr/bin/env python

import setuptools

LICENSE = 'MIT'

if __name__ == "__main__":
    setuptools.setup(
        name='pykernels',
        version='0.0.4',

        description='Python library for working with kernel methods in machine learning',
        author='Wojciech Marian Czarnecki and Katarzyna Janocha',
        author_email='wojciech.czarnecki@uj.edu.pl',
        url='https://github.com/gmum/pykernels',

        license=LICENSE,
        packages=setuptools.find_packages(),

        install_requires=[
            'numpy',
            'scipy',
            'scikit-learn',
        ],

        classifiers=[
            'Development Status :: 1 - Alpha',
            'Intended Audience ::  Machine Learning Research',
            'License :: OSI Approved ::' + LICENSE,
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Machine Learning',
            'Topic :: Scientific/Engineering :: Information Analysis',
        ],

        zip_safe=True,
        include_package_data=True,
    )
