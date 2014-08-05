/*
  C implementation of SIML LINGOs on CPUs

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

inline int32_t max(int32_t a,int32_t b) {
    return a>b ? a : b;
}
inline int32_t min(int32_t a,int32_t b) {
    return a<b?a:b;
}

static float multisetTanimoto3_mags(int32_t* a,int32_t* b,int32_t* asize,int32_t* bsize,int32_t alen,int32_t blen,int32_t amag,int32_t bmag) {
    // Version of Tanimoto code that uses explicit branches
    if (amag == 0 || bmag == 0) return 0.0f;
    int32_t i=0,j=0;
    int32_t isct=0;

    while ( i < alen && j < blen) {
        if (a[i] == b[j]) {
            isct += min(asize[i],bsize[j]);
            i++;
            j++;
        } else if (a[i] < b[j]) {
            i++;
        } else { // b[j] < a[i]
            j++;
        }
    }
    return isct/((float)amag+bmag-isct);
}

#define CHECKARRAYTYPE(ary,name) if (PyArray_TYPE(ary) != NPY_INT32) {\
                                     PyErr_SetString(PyExc_ValueError,name" was not of type int32");\
                                     return NULL;\
                                 } 
#define CHECKARRAYCARRAY(ary,name) if ((PyArray_FLAGS(ary) & NPY_CARRAY) != NPY_CARRAY) {\
                                       PyErr_SetString(PyExc_ValueError,name" was not a contiguous well-behaved array in C order");\
                                       return NULL;\
                                   } 


static PyObject *_CPULingo_getTanimotoBlock(PyObject *self, PyObject *args) {
  
    npy_intp dim2[2];
    int32_t *reflingos,*refcounts,*refmags,*reflengths;
    int32_t *qlingos,*qcounts,*qmags,*qlengths;
    npy_intp *rldims,*rcdims,*rlstrides,*rcstrides,*qldims,*qcdims,*qlstrides,*qcstrides,*tan_strides;
    int nrefmols,nqmols;
    PyArrayObject* ary_tanimotos;
    float* tanimotos;
    int32_t *reflingoset,*refcountset;
    int32_t refmag,reflength;
    float* outputrow;
    int32_t *qlingoset,*qcountset;
    int32_t qmag,qlength;
    int row,col;
    float t;
    int nprocs=1;
  
    PyArrayObject *ary_reflingos,*ary_refcounts,*ary_refmags,*ary_reflengths,*ary_qlingos,*ary_qcounts,*ary_qmags,*ary_qlengths;
  
    if (!PyArg_ParseTuple(args, "OOOOOOOO|i",
              &ary_reflingos, &ary_refcounts, &ary_refmags, &ary_reflengths,
              &ary_qlingos,   &ary_qcounts,   &ary_qmags,   &ary_qlengths,
              &nprocs)) {
      return NULL;
    }

    // This is a serial function. We only accept the argument so that we can
    // support the interface for gTBParallel if the user did not enable OpenMP
    if (nprocs != 1) {
        //fprintf(stderr,"Warning: called _CPULingo.getTanimotoBlocks or getTanimotoBlocksParallel requesting more than one CPU and pysiml not built with OpenMP support. Only using one CPU.\n");
        nprocs = 1;
    }
  
    // Get pointers to array data
    reflingos  = (int32_t*) PyArray_DATA(ary_reflingos);
    refcounts  = (int32_t*) PyArray_DATA(ary_refcounts);
    refmags    = (int32_t*) PyArray_DATA(ary_refmags);
    reflengths = (int32_t*) PyArray_DATA(ary_reflengths);
    qlingos    = (int32_t*) PyArray_DATA(ary_qlingos);
    qcounts    = (int32_t*) PyArray_DATA(ary_qcounts);
    qmags      = (int32_t*) PyArray_DATA(ary_qmags);
    qlengths   = (int32_t*) PyArray_DATA(ary_qlengths);

    // Get dimensions of arrays (# molecules, maxlingos)
    rldims = PyArray_DIMS(ary_reflingos);
    rlstrides = PyArray_STRIDES(ary_reflingos);
    rcdims = PyArray_DIMS(ary_refcounts);
    rcstrides = PyArray_STRIDES(ary_refcounts);
    qldims = PyArray_DIMS(ary_qlingos);
    qlstrides = PyArray_STRIDES(ary_qlingos);
    qcdims = PyArray_DIMS(ary_qcounts);
    qcstrides = PyArray_STRIDES(ary_qcounts);

    // Do some sanity checking on array dimensions {{{
    //      - make sure they are of int32 data type
    CHECKARRAYTYPE(ary_reflingos,"Reference Lingo matrix");
    CHECKARRAYTYPE(ary_refcounts,"Reference Lingo count matrix");
    CHECKARRAYTYPE(ary_refmags,"Reference magnitude vector");
    CHECKARRAYTYPE(ary_reflengths,"Reference length vector");
    CHECKARRAYTYPE(ary_qlingos,"Query Lingo matrix");
    CHECKARRAYTYPE(ary_qcounts,"Query Lingo count matrix");
    CHECKARRAYTYPE(ary_qmags,"Query magnitude vector");
    CHECKARRAYTYPE(ary_qlengths,"Query length vector");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_reflingos->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Reference Lingo matrix did not have dimension 2");
        return NULL;
    }
    if (ary_refcounts->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Reference Lingo count matrix did not have dimension 2");
        return NULL;
    }
    if (rldims[0] != rcdims[0] || rldims[1] != rcdims[1]) {
        PyErr_SetString(PyExc_ValueError,"Reference Lingo and Lingo count matrix did not have identical shapes");
        return NULL;
    }
    if (ary_qlingos->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Query Lingo matrix did not have dimension 2");
        return NULL;
    }
    if (ary_qcounts->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Query Lingo count matrix did not have dimension 2");
        return NULL;
    }
    if (qldims[0] != qcdims[0] || qldims[1] != qcdims[1]) {
        PyErr_SetString(PyExc_ValueError,"Query Lingo and Lingo count matrix did not have identical shapes");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_reflingos,"Reference Lingo matrix");
    CHECKARRAYCARRAY(ary_refcounts,"Reference Lingo count matrix");
    CHECKARRAYCARRAY(ary_refmags,"Reference magnitude vector");
    CHECKARRAYCARRAY(ary_reflengths,"Reference length vector");
    CHECKARRAYCARRAY(ary_qlingos,"Query Lingo matrix");
    CHECKARRAYCARRAY(ary_qcounts,"Query Lingo count matrix");
    CHECKARRAYCARRAY(ary_qmags,"Query magnitude vector");
    CHECKARRAYCARRAY(ary_qlengths,"Query length vector");

    //      - make sure lengths/mags are 1d or (Nx1) and have same length as #rows of l/c arrays
    if (!(ary_reflengths->nd == 1 || (ary_reflengths->nd == 2 && ary_reflengths->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_ValueError,"Reference length vector was not 1-D");
        return NULL;
    }
    if (ary_reflengths->dimensions[0] != rldims[0]) {
        PyErr_SetString(PyExc_ValueError,"Reference length vector length did not equal number of rows of reference Lingo matrix");
        return NULL;
    }
    if (!(ary_refmags->nd == 1 || (ary_refmags->nd == 2 && ary_refmags->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_ValueError,"Reference magnitude vector was not 1-D");
        return NULL;
    }
    if (ary_refmags->dimensions[0] != rldims[0]) {
        PyErr_SetString(PyExc_ValueError,"Reference magnitude vector length did not equal number of rows of reference Lingo matrix");
        return NULL;
    }
    if (!(ary_qlengths->nd == 1 || (ary_qlengths->nd == 2 && ary_qlengths->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_ValueError,"Query length vector was not 1-D");
        return NULL;
    }
    if (ary_qlengths->dimensions[0] != qldims[0]) {
        PyErr_SetString(PyExc_ValueError,"Query length vector length did not equal number of rows of query Lingo matrix");
        return NULL;
    }
    if (!(ary_qmags->nd == 1 || (ary_qmags->nd == 2 && ary_qmags->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_ValueError,"Query magnitude vector was not 1-D");
        return NULL;
    }
    if (ary_qmags->dimensions[0] != qldims[0]) {
        PyErr_SetString(PyExc_ValueError,"Query magnitude vector length did not equal number of rows of query Lingo matrix");
        return NULL;
    }
    // }}}
    
    /*
    printf("Got reference matrix of size %ld x %ld and stride (%ld,%ld)\n",rldims[0],rldims[1],rlstrides[0],rlstrides[1]);
    printf("Got reference lengths of size %ld and stride %ld\n",PyArray_DIMS(ary_reflengths)[0],PyArray_STRIDES(ary_reflengths)[0]);
    printf("Got reference mags of size %ld and stride %ld\n",PyArray_DIMS(ary_refmags)[0],PyArray_STRIDES(ary_refmags)[0]);
    printf("Got query matrix of size %ld x %ld and stride (%ld,%ld)\n",qldims[0],qldims[1],qlstrides[0],qlstrides[1]);
    printf("Got query lengths of size %ld and stride %ld\n",PyArray_DIMS(ary_qlengths)[0],PyArray_STRIDES(ary_qlengths)[0]);
    printf("Got query mags of size %ld and stride %ld\n",PyArray_DIMS(ary_qmags)[0],PyArray_STRIDES(ary_qmags)[0]);
    */

    nrefmols = rldims[0];
    nqmols = qldims[0];

    // Create return array containing Tanimotos
    dim2[0] = nrefmols;
    dim2[1] = nqmols;
    ary_tanimotos = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_FLOAT);
    tanimotos = (float*) PyArray_DATA(ary_tanimotos);
    tan_strides = PyArray_STRIDES(ary_tanimotos);
  
    
    // Fill this array with Tanimotos, one element at a time...
    for (row = 0; row < nrefmols; row++) {
        reflingoset = reflingos + row*rlstrides[0]/4;
        refcountset = refcounts + row*rcstrides[0]/4;
        refmag = refmags[row];
        reflength = reflengths[row];
        outputrow = tanimotos + row*tan_strides[0]/4;
        //printf("Got reference set Lingos:");
        //for (i = 0; i < reflength; i++) printf(" %08x",reflingoset[i]);
        //printf("\n");
        //printf("Got reference set counts:");
        //for (i = 0; i < reflength; i++) printf(" %08x",refcountset[i]);
        //printf("\n");
        //printf("Got reference set length %d, magnitude %d\n",reflength,refmag);

        for (col = 0; col < nqmols; col++) {
            qlingoset = qlingos + col*qlstrides[0]/4;
            qcountset = qcounts + col*qcstrides[0]/4;
            qmag = qmags[col];
            qlength = qlengths[col];
            
            //printf("\tGot query set Lingos:");
            //for (i = 0; i < qlength; i++) printf(" %08x",qlingoset[i]);
            //printf("\n");
            //printf("\tGot query set counts:");
            //for (i = 0; i < qlength; i++) printf(" %08x",qcountset[i]);
            //printf("\n");
            //printf("\tGot query set length %d, magnitude %d\n",qlength,qmag);

            t = multisetTanimoto3_mags(reflingoset,qlingoset,refcountset,qcountset,reflength,qlength,refmag,qmag);
            outputrow[col] = t;
            //printf("\tTanimoto = %f\n",t);
        }
    }

    return PyArray_Return(ary_tanimotos);
}

#ifdef USE_OPENMP
static PyObject *_CPULingo_getTanimotoBlockParallel(PyObject *self, PyObject *args) {
  
    npy_intp dim2[2];
    int32_t *reflingos,*refcounts,*refmags,*reflengths;
    int32_t *qlingos,*qcounts,*qmags,*qlengths;
    npy_intp *rldims,*rcdims,*rlstrides,*rcstrides,*qldims,*qcdims,*qlstrides,*qcstrides,*tan_strides;
    int nrefmols,nqmols;
    PyArrayObject* ary_tanimotos;
    float* tanimotos;
    int32_t *reflingoset,*refcountset;
    int32_t refmag,reflength;
    float* outputrow;
    int32_t *qlingoset,*qcountset;
    int32_t qmag,qlength;
    int row,col;
    float t;
    int nprocs=0;

  
    PyArrayObject *ary_reflingos,*ary_refcounts,*ary_refmags,*ary_reflengths,*ary_qlingos,*ary_qcounts,*ary_qmags,*ary_qlengths;
  
    if (!PyArg_ParseTuple(args, "OOOOOOOO|i",
              &ary_reflingos, &ary_refcounts, &ary_refmags, &ary_reflengths,
              &ary_qlingos,   &ary_qcounts,   &ary_qmags,   &ary_qlengths,
              &nprocs)) {
      return NULL;
    }
  
    // Get pointers to array data
    reflingos  = (int32_t*) PyArray_DATA(ary_reflingos);
    refcounts  = (int32_t*) PyArray_DATA(ary_refcounts);
    refmags    = (int32_t*) PyArray_DATA(ary_refmags);
    reflengths = (int32_t*) PyArray_DATA(ary_reflengths);
    qlingos    = (int32_t*) PyArray_DATA(ary_qlingos);
    qcounts    = (int32_t*) PyArray_DATA(ary_qcounts);
    qmags      = (int32_t*) PyArray_DATA(ary_qmags);
    qlengths   = (int32_t*) PyArray_DATA(ary_qlengths);

    // Get dimensions of arrays (# molecules, maxlingos)
    rldims = PyArray_DIMS(ary_reflingos);
    rlstrides = PyArray_STRIDES(ary_reflingos);
    rcdims = PyArray_DIMS(ary_refcounts);
    rcstrides = PyArray_STRIDES(ary_refcounts);
    qldims = PyArray_DIMS(ary_qlingos);
    qlstrides = PyArray_STRIDES(ary_qlingos);
    qcdims = PyArray_DIMS(ary_qcounts);
    qcstrides = PyArray_STRIDES(ary_qcounts);

    // Do some sanity checking on array dimensions {{{
    //      - make sure they are of int32 data type
    CHECKARRAYTYPE(ary_reflingos,"Reference Lingo matrix");
    CHECKARRAYTYPE(ary_refcounts,"Reference Lingo count matrix");
    CHECKARRAYTYPE(ary_refmags,"Reference magnitude vector");
    CHECKARRAYTYPE(ary_reflengths,"Reference length vector");
    CHECKARRAYTYPE(ary_qlingos,"Query Lingo matrix");
    CHECKARRAYTYPE(ary_qcounts,"Query Lingo count matrix");
    CHECKARRAYTYPE(ary_qmags,"Query magnitude vector");
    CHECKARRAYTYPE(ary_qlengths,"Query length vector");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_reflingos->nd != 2) {
        PyErr_SetString(PyExc_TypeError,"Reference Lingo matrix did not have dimension 2");
        return NULL;
    }
    if (ary_refcounts->nd != 2) {
        PyErr_SetString(PyExc_TypeError,"Reference Lingo count matrix did not have dimension 2");
        return NULL;
    }
    if (rldims[0] != rcdims[0] || rldims[1] != rcdims[1]) {
        PyErr_SetString(PyExc_TypeError,"Reference Lingo and Lingo count matrix did not have identical shapes");
        return NULL;
    }
    if (ary_qlingos->nd != 2) {
        PyErr_SetString(PyExc_TypeError,"Query Lingo matrix did not have dimension 2");
        return NULL;
    }
    if (ary_qcounts->nd != 2) {
        PyErr_SetString(PyExc_TypeError,"Query Lingo count matrix did not have dimension 2");
        return NULL;
    }
    if (qldims[0] != qcdims[0] || qldims[1] != qcdims[1]) {
        PyErr_SetString(PyExc_TypeError,"Query Lingo and Lingo count matrix did not have identical shapes");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_reflingos,"Reference Lingo matrix");
    CHECKARRAYCARRAY(ary_refcounts,"Reference Lingo count matrix");
    CHECKARRAYCARRAY(ary_refmags,"Reference magnitude vector");
    CHECKARRAYCARRAY(ary_reflengths,"Reference length vector");
    CHECKARRAYCARRAY(ary_qlingos,"Query Lingo matrix");
    CHECKARRAYCARRAY(ary_qcounts,"Query Lingo count matrix");
    CHECKARRAYCARRAY(ary_qmags,"Query magnitude vector");
    CHECKARRAYCARRAY(ary_qlengths,"Query length vector");

    //      - make sure lengths/mags are 1d or (Nx1) and have same length as #rows of l/c arrays
    if (!(ary_reflengths->nd == 1 || (ary_reflengths->nd == 2 && ary_reflengths->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_TypeError,"Reference length vector was not 1-D");
        return NULL;
    }
    if (ary_reflengths->dimensions[0] != rldims[0]) {
        PyErr_SetString(PyExc_TypeError,"Reference length vector length did not equal number of rows of reference Lingo matrix");
        return NULL;
    }
    if (!(ary_refmags->nd == 1 || (ary_refmags->nd == 2 && ary_refmags->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_TypeError,"Reference magnitude vector was not 1-D");
        return NULL;
    }
    if (ary_refmags->dimensions[0] != rldims[0]) {
        PyErr_SetString(PyExc_TypeError,"Reference magnitude vector length did not equal number of rows of reference Lingo matrix");
        return NULL;
    }
    if (!(ary_qlengths->nd == 1 || (ary_qlengths->nd == 2 && ary_qlengths->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_TypeError,"Query length vector was not 1-D");
        return NULL;
    }
    if (ary_qlengths->dimensions[0] != qldims[0]) {
        PyErr_SetString(PyExc_TypeError,"Query length vector length did not equal number of rows of query Lingo matrix");
        return NULL;
    }
    if (!(ary_qmags->nd == 1 || (ary_qmags->nd == 2 && ary_qmags->dimensions[1] == 1))) {
        PyErr_SetString(PyExc_TypeError,"Query magnitude vector was not 1-D");
        return NULL;
    }
    if (ary_qmags->dimensions[0] != qldims[0]) {
        PyErr_SetString(PyExc_TypeError,"Query magnitude vector length did not equal number of rows of query Lingo matrix");
        return NULL;
    }
    // }}}
    
    /*
    printf("Got reference matrix of size %ld x %ld and stride (%ld,%ld)\n",rldims[0],rldims[1],rlstrides[0],rlstrides[1]);
    printf("Got reference lengths of size %ld and stride %ld\n",PyArray_DIMS(ary_reflengths)[0],PyArray_STRIDES(ary_reflengths)[0]);
    printf("Got reference mags of size %ld and stride %ld\n",PyArray_DIMS(ary_refmags)[0],PyArray_STRIDES(ary_refmags)[0]);
    printf("Got query matrix of size %ld x %ld and stride (%ld,%ld)\n",qldims[0],qldims[1],qlstrides[0],qlstrides[1]);
    printf("Got query lengths of size %ld and stride %ld\n",PyArray_DIMS(ary_qlengths)[0],PyArray_STRIDES(ary_qlengths)[0]);
    printf("Got query mags of size %ld and stride %ld\n",PyArray_DIMS(ary_qmags)[0],PyArray_STRIDES(ary_qmags)[0]);
    */

    nrefmols = rldims[0];
    nqmols = qldims[0];

    // Create return array containing Tanimotos
    dim2[0] = nrefmols;
    dim2[1] = nqmols;
    ary_tanimotos = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_FLOAT);
    tanimotos = (float*) PyArray_DATA(ary_tanimotos);
    tan_strides = PyArray_STRIDES(ary_tanimotos);
  
    
    // Fill this array with Tanimotos, parallelized over rows
    if (nprocs > 0) omp_set_num_threads(nprocs);

    #pragma omp parallel for default(none) shared(nrefmols,nqmols,rlstrides,rcstrides,reflingos,refcounts,refmags,reflengths,tanimotos,tan_strides,qlingos,qlstrides,qcounts,qcstrides,qmags,qlengths) private(row,col,reflingoset,refcountset,refmag,reflength,qlingoset,qcountset,qmag,qlength,t,outputrow)
    for (row = 0; row < nrefmols; row++) {
        reflingoset = reflingos + row*rlstrides[0]/4;
        refcountset = refcounts + row*rcstrides[0]/4;
        refmag = refmags[row];
        reflength = reflengths[row];
        outputrow = tanimotos + row*tan_strides[0]/4;
        //printf("Got reference set Lingos:");
        //for (i = 0; i < reflength; i++) printf(" %08x",reflingoset[i]);
        //printf("\n");
        //printf("Got reference set counts:");
        //for (i = 0; i < reflength; i++) printf(" %08x",refcountset[i]);
        //printf("\n");
        //printf("Got reference set length %d, magnitude %d\n",reflength,refmag);

        for (col = 0; col < nqmols; col++) {
            qlingoset = qlingos + col*qlstrides[0]/4;
            qcountset = qcounts + col*qcstrides[0]/4;
            qmag = qmags[col];
            qlength = qlengths[col];
            
            //printf("\tGot query set Lingos:");
            //for (i = 0; i < qlength; i++) printf(" %08x",qlingoset[i]);
            //printf("\n");
            //printf("\tGot query set counts:");
            //for (i = 0; i < qlength; i++) printf(" %08x",qcountset[i]);
            //printf("\n");
            //printf("\tGot query set length %d, magnitude %d\n",qlength,qmag);

            t = multisetTanimoto3_mags(reflingoset,qlingoset,refcountset,qcountset,reflength,qlength,refmag,qmag);
            outputrow[col] = t;
            //printf("\tTanimoto = %f\n",t);
        }
    }

    return PyArray_Return(ary_tanimotos);
}
static PyObject *_CPULingo_supportsParallel(PyObject *self, PyObject *args) {
    return Py_True;
}
#else
static PyObject *_CPULingo_supportsParallel(PyObject *self, PyObject *args) {
    return Py_False;
}
#endif

static PyMethodDef _CPULingo_methods[] = {
  {"getTanimotoBlock", (PyCFunction)_CPULingo_getTanimotoBlock, METH_VARARGS, "Computes a block of Tanimotos using the sparse-vector SIML algorithm"},
  {"supportsParallel", (PyCFunction)_CPULingo_supportsParallel, METH_VARARGS, "Returns True if pySIML was built with OpenMP support"},
  #ifdef USE_OPENMP
  {"getTanimotoBlockParallel", (PyCFunction)_CPULingo_getTanimotoBlockParallel, METH_VARARGS, "Computes a block of Tanimotos using the sparse-vector SIML algorithm, parallelized over rows"},
  #else
  {"getTanimotoBlockParallel", (PyCFunction)_CPULingo_getTanimotoBlock, METH_VARARGS, "Computes a block of Tanimotos using the sparse-vector SIML algorithm (warning: pysiml built without OpenMP support, this function is not parallelized)"},
  #endif
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_CPULingo(void)
{
  Py_InitModule3("_CPULingo", _CPULingo_methods, "Computes LINGO Tanimotos using the SIML method\n");   
  import_array();
}
