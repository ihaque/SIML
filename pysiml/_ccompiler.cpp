/*
  C implementation of SIML preprocessor to transform SMILES to sparse-matrix representation

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
#include <string.h>
#include <vector>
using namespace std;


class LingoBag {
    void insert(int32_t lingo) {
        // Insert a new lingo int32_to the sorted set of unique lingos
        int32_t i;
        for (i = 0; i < totallingos; i++ ) {
            if (uniqlingos[i] >= lingo) break;
        }
        // Termination: current element is at insertion point32_t (possibly end) or is equal to element
        if (i == totallingos || uniqlingos[i] != lingo) {
            // Must do an insertion
            // Shift everything to the right
            for (int32_t j = totallingos; j > i; j--) {
                uniqlingos[j] = uniqlingos[j-1];
                lingocounts[j] = lingocounts[j-1];
            }
            uniqlingos[i] = lingo;
            lingocounts[i] = 1;
            totallingos++;
        } else {
            // Found duplicate
            lingocounts[i]++;
        }
        /*char clingo[5];
        clingo[4] = 0;
        ((int32_t*)clingo)[0] = lingo;
        printf("After adding %s to set:\n",clingo);
        for (int32_t i = 0; i < totallingos; i++) {
            ((int32_t*)clingo)[0] = uniqlingos[i];
            printf("\t%08x %s x %d\n",uniqlingos[i],clingo,lingocounts[i]);
        }*/
    }
    public:
    int32_t* uniqlingos;
    int32_t* lingocounts;
    int32_t totallingos;
    int32_t magnitude;
    LingoBag(const char* smiles) {
        //int32_t smilen = strlen(smiles+4);
        int32_t smilen = ((int32_t*)smiles)[0];
        int32_t nblocks = smilen/4 + 1;
        int32_t* lingos = (int32_t*)smiles;
        totallingos = 0;
        // Upper bound on the number of LINGOs
        magnitude = smilen-3;
        if (magnitude < 1) {
            uniqlingos  = NULL;
            lingocounts = NULL;
            totallingos = 0;
            magnitude   = 0;
            return;
        }
        uniqlingos = new int32_t[smilen-3];
        lingocounts = new int32_t[smilen-3];
        memset(lingocounts,0,(smilen-3)*sizeof(int32_t));
        //printf("      SMILES: ");
        //for (int32_t i = 0; smiles[i+4]; i++) printf("%c",smiles[i+4]);
        //printf("\n");
        //printf("Block SMILES: ");
        //for (int32_t i = 0; ((char*)(lingos+1))[i]; i++) printf("%c",((char*)(lingos+1))[i]);
        //printf("\n");
        
        //printf("      SMILES: ");
        //for (int32_t i = 0; i < nblocks; i++) printf("%08x ",((int32_t*)(smiles+4))[i]);
        //printf("\n");
        //printf("Block SMILES: ");
        //for (int32_t i = 0; i < nblocks; i++) printf("%08x ",lingos[i+1]);
        //printf("\n");
        //printf("%d total blocks at end of initialization\n",lingos[0]);

        //printf("LINGOs:\n");
        char lingo[5];
        lingo[4] = 0;
        for (int32_t shift = 0; shift < 4; shift++) {
            for (int32_t i = 0; i < nblocks; i++) {
                ((int32_t*)lingo)[0] = lingos[i+1];
                #ifdef BIGENDIAN
                if (!lingo[0]) break;
                #else
                if (!lingo[3]) break; 
                #endif
                //printf("\t%d: %s\n",shift,lingo);
                insert(lingos[i+1]);
            }
            for (int32_t i = 0; i < nblocks; i++) {
                if (i < nblocks - 1)
                    #ifdef BIGENDIAN
                    lingos[i+1] = lingos[i+1] << 8 | lingos[i+2] >> 24;
                    #else
                    lingos[i+1] = lingos[i+1] >> 8 | ((lingos[i+2] & 0xFF)<<24);
                    #endif
                else
                    #ifdef BIGENDIAN
                    lingos[i+1] <<= 8;
                    #else
                    lingos[i+1] >>= 8;
                    #endif
            }
            //printf("      SMILES: ");
            //for (int32_t i = 0; i < nblocks; i++) printf("%08x ",((int32_t*)smiles)[i]);
            //printf("\n");
            //printf("Block SMILES: ");
            //for (int32_t i = 0; i < nblocks; i++) printf("%08x ",lingos[i+1]);
            //printf("\n");
        }
        //printf("Sorted list of LINGOs:\n");
        //for (int32_t i = 0; i < totallingos; i++) {
        //    ((int32_t*)lingo)[0] = uniqlingos[i];
        //    printf("\t%08x %s x %d\n",uniqlingos[i],lingo,lingocounts[i]);
        //}
        //int mag = 0;
        //for (int i = 0; i < totallingos; i++) {
        //    mag += lingocounts[i];
        //}
        //printf("Magnitude = %d, smilen-3 = %d\n",mag,smilen-3);
        //printf("Total unique %d\n",totallingos);
        

        return;       
    }
    ~LingoBag() {
        if (uniqlingos){
            //printf("Deleting uniqlingos\n");
            delete[] uniqlingos;
        }
        if (lingocounts) {
            //printf("Deleting lingocounts\n"); 
            delete[] lingocounts;
        }
    }
};

int processSMILESdigits(const char* buf,char* out) {
    enum states {NORMAL,CHOMP,REPLACE,SPECIALNUMS,NAME,DONE};
    //char** stringstates[] = {"NORMAL","CHOMP","REPLACE","SPECIALNUMS","NAME","DONE"};
    int state = NORMAL;
    int length = 0;
    const char* inp = buf;
    char* outp = out;
    while (*inp && state != DONE) {
        // Update state
        //printf("Starting character: %c, state = %s, output = '%s'\n",*inp,stringstates[state],out);
        switch (state) {
            case NAME:
                *outp = *inp;
                outp++;
                inp++;
                break;
            case CHOMP:
                inp++;
                state = REPLACE;
                break;
            case REPLACE:
                if (isdigit(*inp)) {
                    *outp='0';
                    length++;
                } else if (*inp == ' ') {
                    //*outp = *inp;
                    //state = NAME
                    *outp = 0;
                    state = DONE;
                } else if (*inp == '\n' || *inp == '\r') {
                    *outp = 0;
                    state = DONE;
                } else if (*inp == '%') {
                    *outp = *inp;
                    state = CHOMP;
                    length++;
                } else {
                    *outp=*inp;
                    state = NORMAL;
                    length++;
                }
                inp++;
                outp++;
                break;
            case SPECIALNUMS:
                *outp = *inp;
                if (!isdigit(*inp)) state = NORMAL;
                length++;
                inp++;
                outp++;
                break;
            case NORMAL:
                if (isdigit(*inp)) {
                    *outp = '0';
                    state = REPLACE;
                    length++;
                } else if (*inp == '%') {
                    *outp = *inp;
                    state = CHOMP;
                    length++;
                } else if (*inp == '+' || *inp == '-' || *inp == 'H' || *inp == '[') {
                    *outp = *inp;
                    state = SPECIALNUMS;
                    length++;
                } else if (*inp == ' ') {
                    //*outp = *inp;
                    //state = NAME;
                    *outp = 0;
                    state = DONE;
                } else if (*inp == '\n' || *inp == '\r') {
                    *outp = 0;
                    state = DONE;
                } else {
                    *outp = *inp;
                    length++;
                }
                inp++;
                outp++;
                break;
            default:
                fprintf(stderr,"ERROR, invalid state reached\n");
        }
    }
    *outp = 0;
    //printf("%s -> %s length (%d)\n",buf,out,length);
    return length;
}

const char* lingoIntToStr(int32_t lingo) {
    static char lingostr[5];
    lingostr[4] = '\0';
    lingostr[0] = (char)(lingo >> 24);
    lingostr[1] = (char)((lingo >> 16) & 0xFF);
    lingostr[2] = (char)((lingo >> 8) & 0xFF);
    lingostr[3] = (char)(lingo & 0xFF);
    return lingostr;

}
static PyObject *_ccompiler_SMILEStoMatrices(PyObject *self, PyObject *args) {
    PyObject *smilist,*pyiterator,*pysmistring;
  
    if (!PyArg_ParseTuple(args, "O",&smilist)) {
      return NULL;
    }
    pyiterator = PyObject_GetIter(smilist);
    
    if (pyiterator == NULL) {
        PyErr_SetString(PyExc_TypeError,"Was not given an iterable object");
        return NULL;
    }

    std::vector<LingoBag*> lbags;
    char* buf = new char[128];
    size_t buflen = 128;
    int32_t maxlingos = 0;
    while ((pysmistring = PyIter_Next(pyiterator))) {
        // Process string
        Py_ssize_t length;
        char* pystr;
        if (PyString_AsStringAndSize(pysmistring,&pystr,&length) == -1) {
            PyErr_SetString(PyExc_TypeError,"Non-string element in sequence");
            Py_DECREF(pysmistring);
            delete[] buf;
            return NULL;
        }
        // Need length of string + NUL terminator + 4 bytes for size + padding to multiple of 4 bytes
        size_t reqbuflen = length + 1 + 4 + ((4 - (length % 4)) % 4);
        if (buflen < reqbuflen) {
            delete[] buf;
            buf = new char[reqbuflen];
            buflen = reqbuflen;
        }

        // Set to zero
        memset(buf,0,reqbuflen);
        
        // Preprocess string (set out to buffer+4), copy in new length
        int32_t newlen = (int32_t) processSMILESdigits(pystr,buf+4);
        ((int32_t*)buf)[0] = newlen;
        
        // Create LingoBag (w/mag!)
        LingoBag* l = new LingoBag(buf);
        
        // Accumulate max length
        maxlingos = max(maxlingos,l->totallingos);
        
        // Add to vector of LBags
        lbags.push_back(l);
        Py_DECREF(pysmistring);

    }
    delete[] buf;

    // Convert LingoBags to ndarrays
    size_t nrows = lbags.size();
    size_t ncols = maxlingos;
    
    npy_intp dim2[2];
    PyArrayObject *ary_lingos,*ary_counts,*ary_lengths,*ary_mags;
    dim2[0] = nrows;
    dim2[1] = ncols;
    ary_lingos = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_INT32);
    ary_counts = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_INT32);
    dim2[1] = 1;
    ary_lengths = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_INT32);
    ary_mags = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_INT32);

    npy_intp *matstrides;
    int32_t *lingos,*counts,*lengths,*mags;
    lingos  = (int32_t*) PyArray_DATA(ary_lingos);
    counts  = (int32_t*) PyArray_DATA(ary_counts);
    lengths = (int32_t*) PyArray_DATA(ary_lengths);
    mags    = (int32_t*) PyArray_DATA(ary_mags);
    // Assume lingo.stride == count.stride
    matstrides = PyArray_STRIDES(ary_lingos);
    
    // Initialize the matrices, since many elements of these will not get
    // touched. Filling with zeros will allow efficient on-disk compressed storage.
    memset(lingos,0,PyArray_NBYTES(ary_lingos));
    memset(counts,0,PyArray_NBYTES(ary_counts));
    
    int row = 0;
    // Accumulate LingoBags into ndarrays
    for (std::vector<LingoBag*>::iterator i = lbags.begin(); i != lbags.end(); i++,row++) {
        lengths[row] = (*i)->totallingos;
        mags[row] = (*i)->magnitude;
        memcpy(lingos+row*matstrides[0]/4,(*i)->uniqlingos, lengths[row]*4);
        memcpy(counts+row*matstrides[0]/4,(*i)->lingocounts,lengths[row]*4);
    }

    // Delete lbags
    for (std::vector<LingoBag*>::iterator i = lbags.begin(); i != lbags.end(); i++) {
        delete *i;
    }

    // Return tuple of ndarrays
    //return PyTuple_Pack(4,ary_lingos,ary_counts,ary_lengths,ary_mags);
    PyObject* rv = PyTuple_New(4);
    PyTuple_SET_ITEM(rv,0,(PyObject*)ary_lingos);
    PyTuple_SET_ITEM(rv,1,(PyObject*)ary_counts);
    PyTuple_SET_ITEM(rv,2,(PyObject*)ary_lengths);
    PyTuple_SET_ITEM(rv,3,(PyObject*)ary_mags);
    return rv;
}

static PyMethodDef _ccompiler_methods[] = {
  {"SMILEStoMatrices", (PyCFunction)_ccompiler_SMILEStoMatrices, METH_VARARGS, "Converts list of SMILES strings to a corresponding set of Numpy matrices/vectors suitable for SIML methods"},
  {NULL, NULL, 0, NULL}
};

extern "C" DL_EXPORT(void) init_ccompiler(void)
{
  Py_InitModule3("_ccompiler", _ccompiler_methods, "C implementation of SMILES->multiset compiler\n");   
  import_array();
}
