#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    name='ttrecipes',
    version='0.1.0',
    description="A cookbook of algorithms and tools that use the tensor train fomat",
    long_description=readme + '\n\n' + history,
    url='https://github.com/rballester/ttrecipes',
    author="Rafael Ballester-Ripoll, Enrique G. Paredes",
    author_email='rballester@ifi.uzh.ch, egparedes@ifi.uzh.ch',
    packages=[
        'ttrecipes',
    ],
    include_package_data=True,
    install_requires=[
        'future',
        'numpy',
        'scipy',
        'ttpy',
        'tabulate'
    ],
    license="LGPL",
    zip_safe=False,
    keywords='ttrecipes',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require='nose'
)
