#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='pykernels',
        version='0.0.4',

        description='Python library for working with kernel methods in machine learning',
        author='Wojciech Marian Czarnecki and Katarzyna Janocha',
        author_email='wojciech.czarnecki@uj.edu.pl',
        url='https://github.com/gmum/pykernels',

        license='MIT',
        packages=setuptools.find_packages(),

        install_requires=[
            'numpy',
            'scipy',
            'scikit-learn',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=True,
        include_package_data=True,
    )
