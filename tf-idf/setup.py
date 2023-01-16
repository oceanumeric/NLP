# compile cython file
import numpy 
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, Extension 


from cos_sim_topn import __version__

ext_utlis = Extension(
    'cos_sim_topn.cos_sim_topn',
    sources=[
        './cos_sim_topn/cos_sim_topn.pyx',
        './cos_sim_topn/sparse_cosine_sim.cpp'
        ],
    include_dirs = [numpy.get_include()],
    extra_compile_args = ['-std=c++0x', '-Os'],
    language='c++',
)

setup(
    name="cos_sim_topn",
    version=__version__,
    setup_requires=[
        'setuptools>=18.0',
        'cython',
    ],
    packages=['cos_sim_topn'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([ext_utlis])
)