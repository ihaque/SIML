#!/usr/bin/env python

#=============================================================================================
# COPYRIGHT NOTICE
#
# Written by Imran S. Haque (ihaque@cs.stanford.edu)
#
# Copyright (c) 2009-2010 Stanford University.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of Stanford University nor the names of its contributors
#       may be used to endorse or promote products derived from this software without
#       specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================================

import sys
import pickle
from optparse import OptionParser
from distutils.core import setup,Extension
import tempfile
import os.path
import distutils.ccompiler

def doSetup(cfg):
    openmp_enabled = cfg.openmp_enabled
    numpy_include_path = cfg.numpy_include_path
    python_include_path = cfg.python_include_path

    cpulingo_extension = None
    compiler_extension = None
    cpuutil_extension = None

    if openmp_enabled:
        cpulingo_compiler_args = ['-fopenmp']
        cpulingo_compiler_defs = [('USE_OPENMP',None)]
        cpulingo_libraries = ['gomp']
    else:
        cpulingo_compiler_args = []
        cpulingo_compiler_defs = []
        cpulingo_libraries = []
    
    cpulingo_extension = Extension('pysiml._CPULingo', ['pysiml/_CPULingo.c'],
                            extra_compile_args = cpulingo_compiler_args, define_macros = cpulingo_compiler_defs, 
                            libraries = cpulingo_libraries, include_dirs = [numpy_include_path,python_include_path])
    compiler_extension = Extension('pysiml._ccompiler',['pysiml/_ccompiler.cpp'],
                            include_dirs = [numpy_include_path,python_include_path])
    cpuutil_extension = Extension('pysiml._cpuutil',['pysiml/_cpuutil.cpp'],
                            include_dirs = [numpy_include_path,python_include_path])
    
    setup(name='pySIML',
          version='1.5',
          description='Python bindings for SIML fast LINGO implementation',
          long_description='Python bindings for SIML fast LINGO implementation, supporting LINGO calculation on CPUs and CUDA-capable GPUs',
          author='Imran Haque',
          author_email='ihaque@cs.stanford.edu',
          url='https://simtk.org/home/siml',
          packages=['pysiml'],
          ext_modules=[cpulingo_extension,
                       compiler_extension,
                       cpuutil_extension],
          license = 'BSD',
          platforms = ['Any'],
          classifiers = [
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Developers',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Operating System :: OS Independent',
                'Topic :: Scientific/Engineering :: Chemistry',
                'Topic :: Software Development :: Libraries :: Python Modules',
                ],
         )

def detectOpenMP():
    compiler = distutils.ccompiler.new_compiler()
    print "Attempting to autodetect OpenMP; ignore anything between the lines."
    print "-------------------------------------------------------------------"
    hasopenmp = compiler.has_function('omp_get_num_threads')
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = compiler.has_function('omp_get_num_threads')
    print "-------------------------------------------------------------------"
    print
    if hasopenmp:
        print "Compiler supports OpenMP, enabling parallel CPULingo.\nIf this is incorrect, you can force-disable OpenMP with configure --disable-openmp"
    else:
        print "Did not detect OpenMP support; disabling parallel CPULingo.\nIf this is incorrect, you can force-enable OpenMP with configure --enable-openmp"
    return hasopenmp

def configureOptions():
    np_include = None
    py_include = None
    try:
        import numpy
        np_include = numpy.get_include()
        print "Autodetected numpy header directory as",np_include,"\nIf this is not correct, please set it manually using the --python-include option to configure"
    except:
        print "Error: I was not able to import numpy. Numpy include path cannot be auto-detected; you may have to set it yourself. Note that pySIML requires numpy to work."

    try:
        import distutils.sysconfig
        py_include = distutils.sysconfig.get_python_inc()
        print "Autodetected Python header directory as",py_include,"\nIf this is not correct, please set it manually using the --numpy-include option to configure"
    except:
        print "Error: I was not able to import distutils.sysconfig to autodetect the Python include directories. If the build fails because it is unable to find Python.h, please re-run configure manually specifying the include directory with the --python-include option"
        
    parser = OptionParser()
    parser.add_option('--numpy-include',action='store',type='string',dest='numpy_include_path',
                      help="Include directory for numpy headers (contains numpy/ directory), if not in default Python directory")
    parser.add_option('--python-include',action='store',type='string',dest='python_include_path',
                      help="Include directory for Python headers (contains Python.h), if not default")
    parser.add_option('--enable-openmp',action='store_true',dest='openmp_enabled',
                      help="Enable OpenMP support for parallel calculation of LINGOs on CPU. Requires OpenMP-capable compiler")
    parser.add_option('--disable-openmp',action='store_false',dest='openmp_enabled',
                      help="Disable OpenMP support for parallel calculation of LINGOs on CPU")
    parser.set_defaults(openmp_enabled=detectOpenMP(),numpy_include_path=np_include,python_include_path=py_include)
    
    # Combine "setup.py" and "configure" in default output if someone runs --help
    scriptname = sys.argv[0]
    sys.argv[0] = " ".join(sys.argv[0:2])

    cfg,args = parser.parse_args(sys.argv[2:])
    # Restore script name
    sys.argv[0] = scriptname
    
    f = open('.siteconfig.pickle','w')
    pickle.dump(cfg,f)
    f.close()
    dumpConfiguration(cfg)
    return cfg

def dumpConfiguration(cfg):
    print
    print "Stored configuration (in .siteconfig.pickle):"
    print "---------------------------------------------"
    print "OpenMP Enabled:",cfg.openmp_enabled
    print "Auxiliary Python include directory:",cfg.python_include_path
    print "Auxiliary Numpy include directory:",cfg.numpy_include_path
    print
    return
    

def checkConfigured():
    try:
        f = open('.siteconfig.pickle','r')
        cfg = pickle.load(f)
        f.close()
        dumpConfiguration(cfg)
        return cfg
    except:
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'configure':
        configureOptions()
    #elif sys.argv[1] == 'sdist':
    #    doSetup()
    else:
        cfg = checkConfigured()
        if not cfg:
            print "Did not detect a configuration. Auto-configuring..."
            cfg = configureOptions()
        doSetup(cfg)
    return
    

if __name__ == "__main__":
    main()
