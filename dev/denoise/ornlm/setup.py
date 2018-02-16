from numpy.distutils.misc_util import get_numpy_include_dirs
import sys
import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

if not 'force_setuptools' in globals():
    # For some always use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_mpkg', 'bdist_wheel', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

# Import distutils _after_ potential setuptools import above, and after removing
# MANIFEST
from distutils.core import setup
from setuptools import Extension

if force_setuptools:
    import setuptools

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
if 'setuptools' in sys.modules:
    from setuptools.dist import Distribution

    Distribution(dict())
else:
    extra_setuptools_args = {}

try:
    from Cython.Distutils.build_ext import build_ext
except ImportError:
    from distutils.command.build_ext import build_ext

cmdclass = {
    'build_ext': build_ext}

setup(
    name='ornlm',
    version='0.1',
    cmdclass=cmdclass,
    ext_modules=[Extension("ornlm", ["ornlm.pyx"],
                           include_dirs=get_numpy_include_dirs(),
                           extra_compile_args=["-msse2 -mfpmath=sse"],
                           language="c++")]
)
#Note on the usage of -msse and -mfpmath-sse compiler options
#http://stackoverflow.com/questions/16888621/why-does-returning-a-floating-point-value-change-its-value
#       -msse2 -mfpmath=sse
