SIML: "Single-Instruction, Multiple LINGOs"
====

Version 1.5

Imran S Haque

12 Jan 2010

https://github.com/ihaque/siml (Forked from SimTK, https://simtk.org/home/siml)

# DESCRIPTION

pySIML is a library implementing an efficient algorithm for the calculation of
LINGO chemical similarities [1]. pySIML includes implementations of its
underlying algorithm for single- and multi-core CPUs and NVIDIA CUDA-enabled
GPUs. It has been tested under both Linux and Mac OS X 10.5.

pySIML is currently distributed in source code form, and is available at
https://github.com/ihaque/siml.

If pySIML or derivative code is used in an academic publication, please cite the
following paper:

    Haque IS, Pande VS, and Walters WP. SIML: A Fast SIMD Algorithm for Calculating LINGO Chemical Similarities on GPUs and CPUs. Journal of Chemical Information and Modeling 2010, 50(4), pp 560-564.


[1] Vidal D, Thormann M, Pons M. LINGO, an Efficient Holographic Text Based
    Method to Calculate Biophysical Properties and Intermolecular Similarities.
    Journal of Chemical Information and Modeling, 2005, 45(2), 386-393.
    DOI:10.1021/ci0496797



# HOW TO BUILD

## Prerequisites

pySIML requires the Python interpreter (has been tested with CPython 2.4 and 2.5),
and the Numpy module; header files for both Python and Numpy are also needed. 

GPU support in pySIML requires the pycuda library (http://mathema.tician.de/software/pycuda)
to be installed. At the time of this writing, the version in the Python Package Index
(0.93) is NOT sufficient to run pySIML; you should check the latest version out from
the Git repository and build and install that (several bugs in GPU context handling have
been fixed since 0.93, but not yet pushed into a new release package).

Parallel CPU LINGO calculation requires that pySIML be built with an OpenMP-capable
C compiler. gcc 4.1 and newer are sufficient; gcc 4.0 is not.

## Configuration

pySIML is distributed as a source tarball using a mostly-standard Python distutils-based
setup procedure. After untarring the package, most people should be able to run:

      python setup.py build
      sudo python setup.py install

In some cases, the setup script will not be able to detect one or more settings
properly, in which case, the configure option can be used:

      python setup.py configure <options>

The following options are available:

        * --enable-openmp: Force pySIML to be built with OpenMP support.
        * --disable-openmp: Force pySIML to be built without OpenMP support.
        * --numpy-include=<dir>: Indicates that the C headers for numpy can be found
          in <dir>. Note that if, for example, arrayobject.h is in 
          /usr/include/python2.5/numpy/arrayobject.h, <dir> should be specified as 
          /usr/include/python2.5, NOT /usr/include/python2.5/numpy.
        * --python-include=<dir>: Indicates that the C headers for Python (e.g., Python.h)
          can be found in <dir>.


# HOW TO RUN

A demonstration program is available in the examples subdirectory as "benchmark.py". Given
a file with SMILES strings (one per line), the benchmark can be run as:

        python examples/benchmark.py [SMILES file] [max # of SMILES to read]

Additional documentation is available in the doc/ directory and on the SimTK site.

# LICENSING

pySIML is licensed under a "New BSD" license, reproduced in the file LICENSE.
