/*
  Utility functions that execute on the CPU

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

*/

#include "Python.h"
#include "numpy/arrayobject.h"
#include <stdint.h>
#include <stdio.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

inline unsigned int min(const unsigned int a,const unsigned int b) {
    return (a < b) ? a : b;
}
inline unsigned int max(const unsigned int a,const unsigned int b) {
    return (a > b) ? a : b;
}
template <class T> void doHistogram(const T* data,const size_t data_rows,const size_t data_cols,
                                    const size_t data_colpitch,const size_t data_rowpitch,
                                    unsigned int* histogram,const size_t hist_colpitch,const size_t hist_rowpitch) {
    const size_t data_eltrowpitch = data_rowpitch / sizeof(T);
    const size_t data_eltcolpitch = data_colpitch / sizeof(T);
    const size_t hist_eltrowpitch = hist_rowpitch / sizeof(unsigned int);
    const size_t hist_eltcolpitch = hist_colpitch / sizeof(unsigned int);

    for (size_t row = 0; row < data_rows; row++) {
        const T* datarow = data + row*data_eltrowpitch;
        unsigned int* histrow = histogram + row*hist_eltrowpitch;
        for (size_t col = 0; col < data_cols; col++) {
            const T* dataelt = datarow + col*data_eltcolpitch;
            unsigned int index = (unsigned int)((*dataelt) * 100);
            index = min(index,100u);
            index = max(index,0u);
            *(histrow + index*hist_eltcolpitch) += 1;
        }
    }
    return;
}

static PyObject *_cpuutil_rowHistogram101(PyObject *self, PyObject *args) {
  
    PyArrayObject *ary_data;
    bool isfloat;
    float* fdata = NULL;
    double* ddata = NULL;
    npy_intp *datadims,*datastrides,*histstrides;
    npy_intp dim2[2];
    PyArrayObject* ary_histogram;
    unsigned int* histogram;
  
  
    if (!PyArg_ParseTuple(args, "O",&ary_data)) {
      return NULL;
    }

    switch (PyArray_TYPE(ary_data)) {
        case NPY_FLOAT32: isfloat = true; break;
        case NPY_FLOAT64: isfloat = false; break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "Data given to rowHistogram101 was not of float (float32) or double (float64) type");
            return NULL;
    }

    // Get pointers to array data
    if (isfloat) fdata = (float*)  PyArray_DATA(ary_data);
    else         ddata = (double*) PyArray_DATA(ary_data);


    // Get dimensions of arrays (# molecules, maxlingos)
    datadims = PyArray_DIMS(ary_data);
    datastrides = PyArray_STRIDES(ary_data);

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_data->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Data given to rowHistogram101 was not of dimension 2");
        return NULL;
    }

    // Create return array containing counts
    dim2[0] = datadims[0];
    dim2[1] = 101;
    //ary_histogram = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_UINT32);
    ary_histogram = (PyArrayObject*) PyArray_ZEROS(2,dim2,NPY_UINT32,0);
    histogram = (unsigned int*) PyArray_DATA(ary_histogram);
    histstrides = PyArray_STRIDES(ary_histogram);
  
    if (isfloat) doHistogram<float>(fdata,datadims[0],datadims[1],datastrides[1],datastrides[0],histogram,histstrides[1],histstrides[0]);
    else        doHistogram<double>(ddata,datadims[0],datadims[1],datastrides[1],datastrides[0],histogram,histstrides[1],histstrides[0]);

    return PyArray_Return(ary_histogram);
}

static PyMethodDef _cpuutil_methods[] = {
  {"rowHistogram101", (PyCFunction)_cpuutil_rowHistogram101, METH_VARARGS, "Histograms each row of a matrix into bins (0,0.01,0.02...,1.0+)"},
  {NULL, NULL, 0, NULL}
};

extern "C" DL_EXPORT(void) init_cpuutil(void)
{
  Py_InitModule3("_cpuutil", _cpuutil_methods, "CPU-based utility functions for SIML");   
  import_array();
}
