#!/usr/bin/python

"""This module provides the user-facing API for calculating LINGO chemical
similarities using the SIML algorithm on CUDA-capable GPUs.
"""

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


import numpy
import time
import sys
import os
import struct
import string
import pycuda.driver as cuda
import pycuda.compiler
import kernels
import pycuda
if pycuda.VERSION[0] == 0 and pycuda.VERSION[1] < 94:
    raise ImportError("Error: pySIML requires pycuda version at least 0.94")

cuda.init()

class _pycudaData: #{{{
    def __init__(self,deviceID,usePycudaArray=False,verboseKernelInfo=False):
        self.device = cuda.Device(deviceID)
        devcaps = self.device.compute_capability()
        self.context = self.device.make_context(cuda.ctx_flags.SCHED_YIELD)
        self.stream = cuda.Stream()

        ksource = [kernels.kernelsource]
        noHistogramming = True
        if devcaps[0] == 1 and devcaps[1] == 1:
            #print "Using SM11 histogramming kernels"
            ksource.append(kernels.sm11_histogramming_kernels)
            noHistogramming = False
        elif (devcaps[0] == 1 and devcaps[1] >= 2) or devcaps[0] > 1:
            #print "Using SM12 histogramming kernels"
            ksource.append(kernels.sm12_histogramming_kernels)
            noHistogramming = False

        self.module = pycuda.compiler.SourceModule("\n".join(ksource))
        self.singleRowKernel = self.module.get_function("gpuLingoSim3_srgf_T_tex2ns")
        self.multiRowKernel = self.module.get_function("gpuLingoSim3_multirow")
        self.nbrKernel = self.module.get_function("gpuGetNeighbors_multirow")
        self.multiRowKernel_nomag = self.module.get_function("gpuLingoSim3_multirow")
        self.refMagKernel   = self.module.get_function("accumulateRefMagnitudes")
        self.qMagKernel     = self.module.get_function("accumulateQueryMagnitudes")
        if verboseKernelInfo:
            print "Single Row Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.singleRowKernel.local_size_bytes,self.singleRowKernel.shared_size_bytes,self.singleRowKernel.num_regs)
            print "Multi Row Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.multiRowKernel.local_size_bytes,self.multiRowKernel.shared_size_bytes,self.multiRowKernel.num_regs)
            print "Neighbor Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.nbrKernel.local_size_bytes,self.nbrKernel.shared_size_bytes,self.nbrKernel.num_regs)
            print "Multi Row Kernel (no magnitudes): %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.multiRowKernel_nomag.local_size_bytes,self.multiRowKernel_nomag.shared_size_bytes,self.multiRowKernel_nomag.num_regs)
            print "Reference Magnitude Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.refMagKernel.local_size_bytes,self.refMagKernel.shared_size_bytes,self.refMagKernel.num_regs)
            print "Query Magnitude Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
             (self.qMagKernel.local_size_bytes,self.qMagKernel.shared_size_bytes,self.qMagKernel.num_regs)
        if noHistogramming:
            self.hist101Kernel         = None
            self.mrLingoHistKernel     = None
        else:
            self.hist101Kernel     = self.module.get_function("histogramMultipleRows101")
            self.mrLingoHistKernel     = self.module.get_function("gpuHistogrammedLingo_multirow")
            if verboseKernelInfo:
                print "101-bin histogramming kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
                 (self.hist101Kernel.local_size_bytes,self.hist101Kernel.shared_size_bytes,self.hist101Kernel.num_regs)
                print "101-bin histogrammed LINGO kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
                 (self.mrLingoHistKernel.local_size_bytes,self.mrLingoHistKernel.shared_size_bytes,self.mrLingoHistKernel.num_regs)


        self.tex2lq = self.module.get_texref("tex2FitLingoMatrixT")
        self.tex2cq = self.module.get_texref("tex2FitLingoCountMatrixT")
        self.tex2lr = self.module.get_texref("tex2RefLingoMatrix")
        self.tex2cr = self.module.get_texref("tex2RefLingoCountMatrix")

        self.rsmiles = None
        self.rcounts = None
        self.rshape = None
        self.qsmiles = None
        self.qcounts = None
        self.ql_gpu = None
        self.rl_gpu = None
        self.qmag_gpu = None
        self.rmag_gpu = None
        self.gpuvector = None
        self.gpumatrix = None # Used to hold results from multiple-row requests

        for tex in (self.tex2lq,self.tex2lr,self.tex2cq,self.tex2cr):
            tex.set_address_mode(0,cuda.address_mode.CLAMP)
            tex.set_address_mode(1,cuda.address_mode.CLAMP)
            if usePycudaArray:
                tex.set_format(cuda.array_format.UNSIGNED_INT32,1)
        return

    def __del__(self):
        # This object may never have acquired a context, so we wrap it in a try/except
        ctx = None
        try:
            ctx = self.context
        except:
            return
        ctx.push()
        del self.stream
        ctx.pop()
        return
        
#}}}

# A decorator function to acquire a CUDA context before the function is called
# and release it afterwards. Cleans up members of GPULingo and prevents me from
# writing context-management bugs...
def _needsContext(f): #{{{
    def wrapped_f(obj,*args,**kwargs):
        obj.gpu.context.push()
        rv = f(obj,*args,**kwargs)
        obj.gpu.context.pop()
        return rv
    wrapped_f.__doc__ = f.__doc__
    return wrapped_f
#}}}


class GPULingo:
    """Object to handle computation of LINGO similarities on GPU with CUDA device ID *deviceid*
    """
    # TODO: possible optimization - check in set_r/qsmiles whether gpu buffers really need to be reallocated
    def __init__(self,deviceID=0): #{{{
        self.qshape = None
        self.nref   = 0
        self.nquery = 0
        self.rlengths = None
        self.usePycudaArray = False
        self.deviceID = deviceID
        self.resultmatrix = None # Used to store host-side results from multiple-row request
        self.resultvector = None
        self.last_async_width = None # Used to store width of last async result (to remove padding)
        self.gpu = _pycudaData(deviceID,self.usePycudaArray)
        self.gpu.context.pop()
    #}}}

    @_needsContext
    def _padded_array(self,ar): #{{{
        nrows_pad = ar.shape[0]
        ncols_pad = 16*((ar.shape[1]+15)/16)
        #arpad = numpy.empty((nrows_pad,ncols_pad),dtype=ar.dtype)
        arpad = cuda.pagelocked_empty((nrows_pad,ncols_pad),dtype=ar.dtype)
        arpad[0:ar.shape[0],0:ar.shape[1]] = ar
        return arpad
    #}}}

    @_needsContext
    def set_refsmiles(self,refsmilesmat,refcountsmat,reflengths,refmags=None): #{{{
        """Sets the reference SMILES set to use Lingo matrix *refsmilesmat*, count matrix *refcountsmat*,
        and length vector *reflengths*. If *refmags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed (on the GPU) from the count matrix.

        Because of hardware limitations, the reference matrices (*refsmilesmat* and *refcountsmat*) must have
        no more than 32,768 rows (molecules) and 65,536 columns (Lingos). Larger computations must be performed in tiles.
        """

        # Set up lingo and count matrices on device #{{{
        if self.usePycudaArray:
            # Set up using PyCUDA CUDAArray support
            self.gpu.rsmiles = cuda.matrix_to_array(refsmilesmat,order='C')
            self.gpu.rcounts = cuda.matrix_to_array(refcountsmat,order='C')
            self.gpu.tex2lr.set_array(self.gpu.rsmiles)
            self.gpu.tex2cr.set_array(self.gpu.rcounts)
        else:
            # Manually handle setup
            temprlmat = self._padded_array(refsmilesmat)
            if temprlmat.shape[1] > 65536 or temprlmat.shape[0] > 32768:
                raise ValueError("Error: reference matrix is not allowed to have more than 64K columns (LINGOs) or 32K rows (molecules) (both padded to multiple of 16). Dimensions = (%d,%d)."%temprlmat.shape)
            self.gpu.rsmiles = cuda.mem_alloc(temprlmat.nbytes)
            cuda.memcpy_htod_async(self.gpu.rsmiles,temprlmat,stream=self.gpu.stream)

            temprcmat = self._padded_array(refcountsmat)
            self.gpu.rcounts = cuda.mem_alloc(temprcmat.nbytes)
            cuda.memcpy_htod_async(self.gpu.rcounts,temprcmat,stream=self.gpu.stream)

            descriptor = cuda.ArrayDescriptor()
            descriptor.width  = temprcmat.shape[1]
            descriptor.height = temprcmat.shape[0]
            descriptor.format = cuda.array_format.UNSIGNED_INT32
            descriptor.num_channels = 1
            self.gpu.tex2lr.set_address_2d(self.gpu.rsmiles,descriptor,temprlmat.strides[0])
            self.gpu.tex2cr.set_address_2d(self.gpu.rcounts,descriptor,temprcmat.strides[0])
            self.gpu.stream.synchronize()
            del temprlmat
            del temprcmat
        #}}}

        self.rlengths = reflengths
        self.rshape = refsmilesmat.shape
        self.nref = refsmilesmat.shape[0]

        # Copy reference lengths to GPU
        self.gpu.rl_gpu = cuda.to_device(reflengths)

        # Allocate buffers for query set magnitudes
        self.gpu.rmag_gpu = cuda.mem_alloc(reflengths.nbytes)
        if refmags is not None:
            cuda.memcpy_htod(self.gpu.rmag_gpu,refmags)
        else:
            # Calculate query set magnitudes on GPU
            magthreads = 256
            self.gpu.refMagKernel(self.gpu.rmag_gpu,self.gpu.rl_gpu,numpy.int32(self.nref),block=(magthreads,1,1),grid=(30,1),shared=magthreads*4,texrefs=[self.gpu.tex2cr])
        return
    #}}}

    @_needsContext
    def set_qsmiles(self,qsmilesmat,qcountsmat,querylengths,querymags=None): #{{{
        """Sets the reference SMILES set to use Lingo matrix *qsmilesmat*, count matrix *qcountsmat*,
        and length vector *querylengths*. If *querymags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed (on the GPU) from the count matrix.

        Because of hardware limitations, the query matrices (*qsmilesmat* and *qcountsmat*) must have
        no more than 65,536 rows (molecules) and 32,768 columns (Lingos). Larger computations must be performed in tiles.
        """
        # Set up lingo and count matrices on device #{{{

        if self.usePycudaArray:
            # Create CUDAarrays for lingo and count matrices
            print "Strides qsmilesmat:",numpy.ascontiguousarray(qsmilesmat.T).strides
            self.gpu.qsmiles = cuda.matrix_to_array(numpy.ascontiguousarray(qsmilesmat.T),order='C')
            self.gpu.qcounts= cuda.matrix_to_array(numpy.ascontiguousarray(qcountsmat.T),order='C')
            print "qsmiles descriptor",dtos(self.gpu.qsmiles.get_descriptor())
            print "qcounts descriptor",dtos(self.gpu.qcounts.get_descriptor())
            self.gpu.tex2lq.set_array(self.gpu.qsmiles)
            self.gpu.tex2cq.set_array(self.gpu.qcounts)
        else:
            # Manually handle texture setup
            # padded_array will handle making matrix contiguous
            tempqlmat = self._padded_array(qsmilesmat.T)
            if tempqlmat.shape[1] > 65536 or tempqlmat.shape[0] > 32768:
                raise ValueError("Error: query matrix is not allowed to have more than 65536 rows (molecules) or 32768 columns (LINGOs) (both padded to multiple of 16). Dimensions = (%d,%d)"%tempqlmat.shape)
            if self.gpu.qsmiles is None or self.gpu.qsmiles.nbytes < tempqlmat.nbytes:
                self.gpu.qsmiles = cuda.mem_alloc(tempqlmat.nbytes)
                self.gpu.qsmiles.nbytes = tempqlmat.nbytes
            cuda.memcpy_htod_async(self.gpu.qsmiles,tempqlmat,stream=self.gpu.stream)

            tempqcmat = self._padded_array(qcountsmat.T)
            if self.gpu.qcounts is None or self.gpu.qcounts.nbytes < tempqcmat.nbytes:
                self.gpu.qcounts = cuda.mem_alloc(tempqcmat.nbytes)
                self.gpu.qcounts.nbytes = tempqcmat.nbytes
            cuda.memcpy_htod_async(self.gpu.qcounts,tempqcmat,stream=self.gpu.stream)
            
            descriptor = cuda.ArrayDescriptor()
            descriptor.width  = tempqcmat.shape[1]
            descriptor.height = tempqcmat.shape[0]
            descriptor.format = cuda.array_format.UNSIGNED_INT32
            descriptor.num_channels = 1
            self.gpu.tex2lq.set_address_2d(self.gpu.qsmiles,descriptor,tempqlmat.strides[0])
            self.gpu.tex2cq.set_address_2d(self.gpu.qcounts,descriptor,tempqcmat.strides[0])
            #print "Set up query textures with stride=",tempqmat.strides[0]
            self.gpu.stream.synchronize()
            del tempqlmat
            del tempqcmat
        #}}}

        self.qshape = qsmilesmat.shape
        self.nquery = qsmilesmat.shape[0]
        #print "Query shape=",self.qshape,", nquery=",self.nquery

        # Transfer query lengths array to GPU
        self.gpu.ql_gpu = cuda.to_device(querylengths)

        # Allocate buffers for query set magnitudes
        self.gpu.qmag_gpu = cuda.mem_alloc(querylengths.nbytes)
        if querymags is not None:
            cuda.memcpy_htod(self.gpu.qmag_gpu,querymags)
        else:
            # Calculate query set magnitudes on GPU
            magthreads = 256
            self.gpu.qMagKernel(self.gpu.qmag_gpu,self.gpu.ql_gpu,numpy.int32(self.nquery),block=(magthreads,1,1),grid=(30,1),shared=magthreads*4,texrefs=[self.gpu.tex2cq])
            #self.qmag_gpu = cuda.to_device(qcountsmat.sum(1).astype(numpy.int32))


        return
    #}}}

    def getTanimotoRow(self,rowidx): #{{{
        """Returns the single Tanimoto row *row* corresponding to comparing every SMILES string
        in the query set with the single reference SMILES string having index *row* in the reference set.

        This method is synchronous (it will not return until the entire row has been computed and brought back from
        the GPU).
        """
        self.getTanimotoRow_async(rowidx)
        return self.retrieveAsyncResult()
    #}}}

    @_needsContext
    def getTanimotoRow_async(self,rowidx): #{{{
        """Compute the single Tanimoto row *row* corresponding to comparing every SMILES string
        in the query set with the single reference SMILES string having index *row* in the reference set, and store it
        as the most recent asynchronous result.

        This method is asynchronous (it will return before the row has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        if self.gpu.rsmiles is None or self.gpu.qsmiles is None:
            raise 

        self.resultmatrix = numpy.empty((1,self.nquery),dtype=numpy.float32)
        self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)

        self.gpu.singleRowKernel.prepare("iPPiii",block=(192,1,1),shared=int(2*4*self.rlengths[rowidx]),\
                                         texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cr,self.gpu.tex2cq])

        self.gpu.singleRowKernel.prepared_async_call((30,1),self.gpu.stream,
                 self.rlengths[rowidx],self.gpu.ql_gpu,self.gpu.gpumatrix,self.qshape[0],self.qshape[1],rowidx)

        self.last_async_width = self.nquery
        self.retrieveAsyncResult = self.retrieveAsyncMatrixResult
        return
    #}}}

    def getMultipleRows(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in
        the reference set, and returns this block of rows.

        This method is synchronous (it will not return until the block has been completely computed).
        """
        self.getMultipleRows_async(rowbase,rowlimit)
        return self.retrieveAsyncResult()
    #}}}

    @_needsContext
    def getMultipleRows_async(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in the reference set,
        and stores this block as the most recent asynchronous result.

        This method is asynchronous (it will return before the block has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        if rowbase < 0 or rowlimit > self.nref:
            raise

        # Pad rows out to 64 byte pitch
        rowpitchInFloat = 16*((self.nquery+15)/16)

        # Using pagelocked memory and async copy seems to actually slow us down
        # on large tiled calculations
        #if self.resultrows is not None and self.resultrows.nbytes != ((rowlimit-rowbase)*rowpitchInFloat*4):
        #    print "Reallocating resultrows..."
        #    del self.resultrows
        #self.resultrows = cuda.pagelocked_empty((rowlimit-rowbase,rowpitchInFloat),dtype=numpy.float32)
        self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInFloat),dtype=numpy.float32)
        self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)

        # With precalculated magnitudes
        self.gpu.multiRowKernel.prepare("PPPPPiiii",block=(192,1,1),shared=int(2*4*max(self.rlengths[rowbase:rowlimit])),\
                                        texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cq,self.gpu.tex2cr])

        self.gpu.multiRowKernel.prepared_async_call((rowlimit-rowbase,1),\
                                                     self.gpu.stream,\
                                                     self.gpu.rl_gpu, self.gpu.ql_gpu, self.gpu.rmag_gpu, self.gpu.qmag_gpu,\
                                                     self.gpu.gpumatrix, rowpitchInFloat, self.qshape[0], self.qshape[1], rowbase)

        self.last_async_width = self.nquery
        self.retrieveAsyncResult = self.retrieveAsyncMatrixResult
        #cuda.memcpy_dtoh_async(self.resultrows,self.gpu.gpurows,stream=self.gpu.stream)
        

        return
    #}}}

    def getMultipleHistogrammedRows(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in
        the reference set. Histograms each row into its own histogram of 101 bins with boundaries 0, 0.01, 
        0.02, ... , 0.99, 1.0, 1.01. Returns this block of row-wise histograms.

        This method is synchronous (it will not return until the histograms have been completely computed).
        """
        self.getMultipleHistogrammedRows_async(rowbase,rowlimit)
        return self.retrieveAsyncResult()
    #}}}

    @_needsContext
    def getMultipleHistogrammedRows_async(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in
        the reference set. Histograms each row into its own histogram of 101 bins with boundaries 0, 0.01, 
        0.02, ... , 0.99, 1.0, 1.01. Returns this block of row-wise histograms.

        This method is asynchronous (it will return before the block has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        if rowbase < 0 or rowlimit > self.nref:
            raise

        # Pad rows out to 64 byte pitch - 112 is smallest multiple of 16 larger than the 101 needed
        rowpitchInInt32 = 112

        # Using pagelocked memory and async copy seems to actually slow us down
        # on large tiled calculations
        #if self.resultrows is not None and self.resultrows.nbytes != ((rowlimit-rowbase)*rowpitchInFloat*4):
        #    print "Reallocating resultrows..."
        #    del self.resultrows
        #self.resultrows = cuda.pagelocked_empty((rowlimit-rowbase,rowpitchInFloat),dtype=numpy.float32)
        self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInInt32),dtype=numpy.int32)
        self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)

        # With precalculated magnitudes
        nthreads = 192
        #shared=int(2*4*max(self.rlengths[rowbase:rowlimit]))+101*4*(nthreads/32)
        #print "Asking for",shared,"bytes of shmem"
        self.gpu.mrLingoHistKernel.prepare("PPPPPiiii",block=(nthreads,1,1),
                                           shared=int(2*4*max(self.rlengths[rowbase:rowlimit]))+101*4*(nthreads/32),\
                                           texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cq,self.gpu.tex2cr])

        self.gpu.mrLingoHistKernel.prepared_async_call((rowlimit-rowbase,1),\
                                                       self.gpu.stream,\
                                                       self.gpu.rl_gpu, self.gpu.ql_gpu, self.gpu.rmag_gpu, self.gpu.qmag_gpu,\
                                                       self.gpu.gpumatrix, rowpitchInInt32, self.qshape[0], self.qshape[1], rowbase)
        #cuda.memcpy_dtoh_async(self.resultrows,self.gpu.gpurows,stream=self.gpu.stream)

        self.last_async_width = 101
        self.retrieveAsyncResult = self.retrieveAsyncMatrixResult
        

        return
    #}}}

    @_needsContext
    def getNeighbors_async(self,rowbase,rowlimit,lowerbound,upperbound=1.1,maxneighbors=None): #{{{
        """For each reference SMILES string with index in *rowbase:rowlimit* (i.e., strings with
        index *row*, *row+1*, ... ,*rowlimit-1*, finds all SMILES in the query set that have
        LINGO similarity >= *lowerbound* and < *upperbound* ("neighbors"), up to a maximum of
        *maxneighbors* (by default, size of query set).

        Result is a tuple of (matrix,vector). The vector contains, for each reference string in
        *rowbase:rowlimit*, the number of neighbors found. The matrix is of size (rowlimit-rowbase,
        maxNeighborsFound), where maxNeighborsFound is the maximum value in the returned vector.
        Each row of the matrix (corresponding to one reference SMILES string) has as its elements
        the query indices of neighbors. In row i, only the first vector[i] elements are valid (that 
        is, the values elements of the matrix beyond the number of neighbors found for a given row 
        are undefined).

        This method is asynchronous (it will return before the block has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        pair may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        if rowbase < 0 or rowlimit > self.nref:
            raise
        if maxneighbors is None:
            maxneighbors = self.nquery

        # Pad rows out to 64 byte pitch
        rowpitchInInt32 = 16*((maxneighbors+15)/16)

        self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInInt32),dtype=numpy.int32)
        #print "Allocated result matrix:",self.resultmatrix.shape
        self.resultvector = numpy.empty((rowlimit-rowbase,),dtype=numpy.int32)
        #print "Allocated result vector:",self.resultvector.shape
        self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)
        self.gpu.gpuvector = cuda.mem_alloc(self.resultvector.nbytes)

        # With precalculated magnitudes
        self.gpu.nbrKernel.prepare("PPPPPiPffiii",block=(128,1,1),shared=int(2*4*max(self.rlengths[rowbase:rowlimit])),\
                                   texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cq,self.gpu.tex2cr])

        self.gpu.nbrKernel.prepared_async_call((rowlimit-rowbase,1),\
                                               self.gpu.stream,\
                                               self.gpu.rl_gpu, self.gpu.ql_gpu, self.gpu.rmag_gpu, self.gpu.qmag_gpu,\
                                               self.gpu.gpumatrix, rowpitchInInt32, self.gpu.gpuvector,
                                               lowerbound,upperbound,
                                               self.qshape[0], self.qshape[1], rowbase)

        self.last_async_width = maxneighbors
        self.retrieveAsyncResult = self.retrieveAsyncMatrixVectorResult_trimmed
        return 
    #}}}

    def getNeighbors(self,rowbase,rowlimit,lowerbound,upperbound=1.1,maxneighbors=None): #{{{
        """For each reference SMILES string with index in *rowbase:rowlimit* (i.e., strings with
        index *row*, *row+1*, ... ,*rowlimit-1*, finds all SMILES in the query set that have
        LINGO similarity >= *lowerbound* and < *upperbound* ("neighbors"), up to a maximum of
        *maxneighbors* (by default, size of query set).

        Result is a tuple of (matrix,vector). The vector contains, for each reference string in
        *rowbase:rowlimit*, the number of neighbors found. The matrix is of size (rowlimit-rowbase,
        maxNeighborsFound), where maxNeighborsFound is the maximum value in the returned vector.
        Each row of the matrix (corresponding to one reference SMILES string) has as its elements
        the query indices of neighbors. In row i, only the first vector[i] elements are valid (that 
        is, the values elements of the matrix beyond the number of neighbors found for a given row 
        are undefined).

        This method is synchronous (it will not return until the neighbors have been completely
        computed. Returns a tuple of (neighborMatrix,neighborCountVector).
        """
        self.getNeighbors_async(rowbase,rowlimit,lowerbound,upperbound,maxneighbors)
        return self.retrieveAsyncResult()
    #}}}

    @_needsContext
    def retrieveAsyncMatrixResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        """

        self.gpu.stream.synchronize()
        
        cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        self.gpu.gpumatrix.free()

        self.retrieveAsyncResult = self.retrieveAsyncResultBase

        return self.resultmatrix[:,0:self.last_async_width]
    #}}}

    @_needsContext
    def retrieveAsyncMatrixVectorResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        UPDATE THIS for neighbors:
        """
        self.gpu.stream.synchronize()
        
        #print "Memcpy size: %d byte matrix, %d byte vector"%(self.resultmatrix.nbytes,self.resultvector.nbytes)
        cuda.memcpy_dtoh(self.resultvector,self.gpu.gpuvector)
        self.gpu.gpuvector.free()
        t = time.time()
        cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        t2 = time.time()
        #print "Achieved %.2f MB/s bandwidth in %d x %d (%d byte) matrix copy"%((self.resultmatrix.nbytes/1048576.0)/(t2-t),self.resultmatrix.shape[0],self.resultmatrix.shape[1],self.resultmatrix.nbytes)
        self.gpu.gpumatrix.free()

        self.retrieveAsyncResult = self.retrieveAsyncResultBase

        return (self.resultmatrix[:,0:self.last_async_width],self.resultvector)
    #}}}

    @_needsContext
    def retrieveAsyncMatrixVectorResult_trimmed(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        UPDATE THIS for neighbors:
        """
        self.gpu.stream.synchronize()
        
        #print "Memcpy size: %d byte matrix, %d byte vector"%(self.resultmatrix.nbytes,self.resultvector.nbytes)
        cuda.memcpy_dtoh(self.resultvector,self.gpu.gpuvector)
        self.gpu.gpuvector.free()

        # We only need to retrieve as many columns as specified by the max of the vector elements
        ncols = numpy.max(self.resultvector)

        matrixcopy = cuda.Memcpy2D()
        elttype = self.resultmatrix.dtype
        matrixcopy.set_src_device(self.gpu.gpumatrix)
        matrixcopy.src_pitch = self.resultmatrix.shape[1]*elttype.itemsize
        self.resultmatrix = numpy.empty((self.resultmatrix.shape[0],ncols),dtype=elttype)
        matrixcopy.set_dst_host(self.resultmatrix)
        matrixcopy.dst_pitch=self.resultmatrix.shape[1]*elttype.itemsize
        matrixcopy.width_in_bytes=self.resultmatrix.shape[1]*elttype.itemsize
        matrixcopy.height=self.resultmatrix.shape[0]
        
        #cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        matrixcopy(True)
        self.gpu.gpumatrix.free()

        self.retrieveAsyncResult = self.retrieveAsyncResultBase

        #return (self.resultmatrix[:,0:self.last_async_width],self.resultvector)
        return (self.resultmatrix,self.resultvector)
    #}}}

    def retrieveAsyncResultBase(self): #{{{
        """TODO
        """
        raise NotImplementedException("Error: tried to call retrieveAsyncResult without a pending asynchronous result")
    #}}}
    def retrieveAsyncResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async`,
        :meth:`getMultipleRows_async`, :meth:`getMultipleHistogrammedRows_async`, or :meth:`getNeighbors_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, result is undefined. If an asynchronous
        operation is still pending, this method will block until completion.
        """
        raise NotImplementedException("Error: tried to call retrieveAsyncResult without a pending asynchronous result")
    #}}}
    
    @_needsContext
    def asyncOperationsDone(self): #{{{
        """Return True if all asynchronous operations on this object have completed.
        """
        return self.gpu.stream.is_done()
    #}}}
