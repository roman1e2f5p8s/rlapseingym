#!/usr/bin/env python

import os, sys
from codecs import open
from setuptools import setup

WORK_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, os.path.join(WORK_DIR, 'rlapse'))
from version import VERSION

with open(os.path.join(WORK_DIR, 'README.md'), mode='r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


# This setup is suitable for "python setup.py develop".
setup(
        name='rlapse',

        version=VERSION,

        description='Reinforcement Learning with Algorithms from Probabilistic Structure Estimation '+\
                '(RLAPSE)',

        author='Roman Overko',

        author_email=['roman.1e2f5p8s.over3@gmail.com', 'roman.overko@ucdconnect.ie'],

        url='https://github.com/roman1e2f5p8s/rlapseingym',

        packages=['rlapse', 'rlapse.mdps', 'rlapse.algorithms', 'rlapse.utils'],

        license='none',

        keywords='reinforcement learning, decision making, machine learning, Markov decision processes',

        install_requires='blackhc.mdp @ git+https://github.com/BlackHC/mdp@67f609f2541090d39957853b59d3affe737d183c',

        tests_require=['pytest'],

        setup_requires=['pytest-runner'],

        classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Libraries :: Python Modules',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3.6',
    ],
)
