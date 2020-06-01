from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler

import numpy as np

# Prepare lbfgs
from Cython.Build import cythonize

class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

include_dirs = ['liblbfgs', np.get_include()]

ext_modules = cythonize(
    [Extension('neurobiases.lbfgs._lowlevel',
               ['neurobiases/lbfgs/_lowlevel.pyx', 'liblbfgs/lbfgs.c'],
               include_dirs=include_dirs)])

setup(
    name='neurobiases',
    version='0.0.0',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    install_requires=[
        'numpy',
        'h5py',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext})
