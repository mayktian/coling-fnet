from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy
extensions= [
		Extension("cwsabie_inner", ["cwsabie_inner.pyx"], include_dirs=[numpy.get_include()]),Extension("transE_label", ["transE_label.pyx"], include_dirs=[numpy.get_include()]),
		Extension("transE_label_triple", ["transE_label_triple.pyx"], include_dirs=[numpy.get_include()]),
		Extension("cython_Wsabie_transE", ["cython_Wsabie_transE.pyx"], include_dirs=[numpy.get_include()]),
		Extension("cwsabie_label", ["cwsabie_label.pyx"], include_dirs=[numpy.get_include()])
]
setup(ext_modules = cythonize(extensions))

