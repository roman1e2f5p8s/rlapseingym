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
                '(RLAPSE) for OpenAI Gym',

        author='Roman Overko',

        author_email='roman.overko@ucdconnect.ie',

        url='https://github.com/roman1e2f5p8s/rlapseingym',

        packages=['rlapse', 'rlapse.mdps', 'rlapse.algorithms', 'rlapse.utils'],

        license='none',

        keywords='reinforcement learning, decision making, machine learning, Markov decision processes',

        tests_require=['pytest'],

        setup_requires=['pytest-runner'],
)
