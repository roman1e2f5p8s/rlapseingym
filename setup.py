#!/usr/bin/env python

from setuptools import setup


# This setup is suitable for "python setup.py develop".
setup(
        name='rlapse',
        version='0.1',
        description='RLAPSE package',
        author='Roman Overko',
        author_email='roman.overko@ucdconnect.ie',
        url='https://github.com/roman1e2f5p8s/rlapseingym',
        packages=['rlapse', 'rlapse.mdps', 'rlapse.algorithms', 'rlapse.utils'],
)
