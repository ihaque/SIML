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
import pyopencl as cl
import clkernels

class _clData: #{{{
    def __init__(self,device,verboseKernelInfo=False):
        self.device = device
        self.context = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.context)

        ksource = [clkernels.kernelsource]
        #noHistogramming = True
        #if devcaps[0] == 1 and devcaps[1] == 1:
        #    #print "Using SM11 histogramming kernels"
        #    ksource.append(kernels.sm11_histogramming_kernels)
        #    noHistogramming = False
        #elif (devcaps[0] == 1 and devcaps[1] >= 2) or devcaps[0] > 1:
        #    #print "Using SM12 histogramming kernels"
        #    ksource.append(kernels.sm12_histogramming_kernels)
        #    noHistogramming = False

        self.module = cl.Program(self.context,"\n".join(ksource)).build()
        #self.singleRowKernel = self.module.get_function("gpuLingoSim3_srgf_T_tex2ns")
        
        #self.multiRowKernel = self.module.gpuLingoSim3_multirow_grgf_T_sync
        self.multiRowKernel = self.module.gpuLingoSim3_multirow_lrgf_T_sync

        #self.nbrKernel = self.module.get_function("gpuGetNeighbors_multirow")
        #self.multiRowKernel_nomag = self.module.get_function("gpuLingoSim3_multirow")
        self.refMagKernel   = self.module.accumulateRefMagnitudes
        #self.qMagKernel     = self.module.get_function("accumulateQueryMagnitudes")
        #if verboseKernelInfo:
        #    print "Single Row Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.singleRowKernel.local_size_bytes,self.singleRowKernel.shared_size_bytes,self.singleRowKernel.num_regs)
        #    print "Multi Row Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.multiRowKernel.local_size_bytes,self.multiRowKernel.shared_size_bytes,self.multiRowKernel.num_regs)
        #    print "Neighbor Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.nbrKernel.local_size_bytes,self.nbrKernel.shared_size_bytes,self.nbrKernel.num_regs)
        #    print "Multi Row Kernel (no magnitudes): %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.multiRowKernel_nomag.local_size_bytes,self.multiRowKernel_nomag.shared_size_bytes,self.multiRowKernel_nomag.num_regs)
        #    print "Reference Magnitude Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.refMagKernel.local_size_bytes,self.refMagKernel.shared_size_bytes,self.refMagKernel.num_regs)
        #    print "Query Magnitude Kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #     (self.qMagKernel.local_size_bytes,self.qMagKernel.shared_size_bytes,self.qMagKernel.num_regs)
        #if noHistogramming:
        #    self.hist101Kernel         = None
        #    self.mrLingoHistKernel     = None
        #else:
        #    self.hist101Kernel     = self.module.get_function("histogramMultipleRows101")
        #    self.mrLingoHistKernel     = self.module.get_function("gpuHistogrammedLingo_multirow")
        #    if verboseKernelInfo:
        #        print "101-bin histogramming kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #         (self.hist101Kernel.local_size_bytes,self.hist101Kernel.shared_size_bytes,self.hist101Kernel.num_regs)
        #        print "101-bin histogrammed LINGO kernel: %d bytes local mem, %d bytes shared mem, %d registers" %\
        #         (self.mrLingoHistKernel.local_size_bytes,self.mrLingoHistKernel.shared_size_bytes,self.mrLingoHistKernel.num_regs)


        #self.tex2lq = self.module.get_texref("tex2FitLingoMatrixT")
        #self.tex2cq = self.module.get_texref("tex2FitLingoCountMatrixT")
        #self.tex2lr = self.module.get_texref("tex2RefLingoMatrix")
        #self.tex2cr = self.module.get_texref("tex2RefLingoCountMatrix")

        self.rsmiles = None
        self.rcounts = None
        self.rshape = None
        #self.qsmiles = None
        #self.qcounts = None
        #self.ql_gpu = None
        self.rl_gpu = None
        #self.qmag_gpu = None
        self.rmag_gpu = None
        #self.gpuvector = None
        #self.gpumatrix = None # Used to hold results from multiple-row requests

        #for tex in (self.tex2lq,self.tex2lr,self.tex2cq,self.tex2cr):
        #    tex.set_address_mode(0,cuda.address_mode.CLAMP)
        #    tex.set_address_mode(1,cuda.address_mode.CLAMP)
        #    if usePycudaArray:
        #        tex.set_format(cuda.array_format.UNSIGNED_INT32,1)
        return

    def __del__(self):
        return
        
#}}}

class OCLLingo:
    """Object to handle computation of LINGO similarities on GPU with CUDA device ID *deviceid*
    """
    # TODO: possible optimization - check in set_r/qsmiles whether gpu buffers really need to be reallocated
    def __init__(self,device): #{{{
        self.qshape = None
        self.nref   = 0
        self.nquery = 0
        self.rlengths = None
        self.device = device
        self.resultmatrix = None # Used to store host-side results from multiple-row request
        self.resultvector = None
        self.last_async_width = None # Used to store width of last async result (to remove padding)
        self.gpu = _clData(device)
    #}}}

    def _padded_array(self,ar): #{{{
        nrows_pad = ar.shape[0]
        ncols_pad = 16*((ar.shape[1]+15)/16)
        #arpad = numpy.empty((nrows_pad,ncols_pad),dtype=ar.dtype)
        arpad = numpy.empty((nrows_pad,ncols_pad),dtype=ar.dtype)
        arpad[0:ar.shape[0],0:ar.shape[1]] = ar
        return arpad
    #}}}

    def set_refsmiles(self,refsmilesmat,refcountsmat,reflengths,refmags=None): #{{{
        """Sets the reference SMILES set to use Lingo matrix *refsmilesmat*, count matrix *refcountsmat*,
        and length vector *reflengths*. If *refmags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed (on the GPU) from the count matrix.

        Because of hardware limitations, the reference matrices (*refsmilesmat* and *refcountsmat*) must have
        no more than 32,768 rows (molecules) and 65,536 columns (Lingos). Larger computations must be performed in tiles.
        """

        # Set up lingo and count matrices on device #{{{
        temprlmat = self._padded_array(refsmilesmat)
        # TODO: use IMAGE2D_MAX_HEIGHT/WIDTH
        if temprlmat.shape[1] > 65536 or temprlmat.shape[0] > 32768:
            raise ValueError("Error: reference matrix is not allowed to have more than 64K columns (LINGOs) or 32K rows (molecules) (both padded to multiple of 16). Dimensions = (%d,%d)."%temprlmat.shape)
        self.gpu.rsmiles = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=temprlmat)

        temprcmat = self._padded_array(refcountsmat)
        self.gpu.rcounts = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=temprcmat)

        self.refPitchInInt = numpy.int32(temprcmat.shape[1])
        #descriptor = cuda.ArrayDescriptor()
        #descriptor.width  = temprcmat.shape[1]
        #descriptor.height = temprcmat.shape[0]
        #descriptor.format = cuda.array_format.UNSIGNED_INT32
        #descriptor.num_channels = 1
        #self.gpu.tex2lr.set_address_2d(self.gpu.rsmiles,descriptor,temprlmat.strides[0])
        #self.gpu.tex2cr.set_address_2d(self.gpu.rcounts,descriptor,temprcmat.strides[0])
        #self.gpu.stream.synchronize()
        del temprlmat
        del temprcmat
        #}}}

        self.rlengths = reflengths
        self.rshape = refsmilesmat.shape
        self.nref = refsmilesmat.shape[0]

        # Copy reference lengths to GPU
        self.gpu.rl_gpu = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=reflengths)

        # Allocate buffers for ref set magnitudes
        self.gpu.rmag_gpu = cl.Buffer(self.gpu.context,cl.mem_flags.READ_WRITE,size=reflengths.nbytes)
        self.gpu.refMagKernel(self.gpu.queue,(64*refcountsmat.shape[0],),
                                              self.gpu.rcounts, self.gpu.rl_gpu, self.gpu.rmag_gpu,self.refPitchInInt,
                                              local_size=(64,))
        hostmags = numpy.empty_like(reflengths)
        cl.enqueue_read_buffer(self.gpu.queue,self.gpu.rmag_gpu,hostmags).wait()
        return hostmags
        #if refmags is not None:
        #    cuda.memcpy_htod(self.gpu.rmag_gpu,refmags)
        #else:
        #    # Calculate ref set magnitudes on GPU
        #    magthreads = 256
        #    self.gpu.refMagKernel(self.gpu.rmag_gpu,self.gpu.rl_gpu,numpy.int32(self.nref),block=(magthreads,1,1),grid=(30,1),shared=magthreads*4,texrefs=[self.gpu.tex2cr])
        return
    #}}}

    def set_qsmiles(self,qsmilesmat,qcountsmat,querylengths,querymags): #{{{
        """Sets the reference SMILES set to use Lingo matrix *qsmilesmat*, count matrix *qcountsmat*,
        and length vector *querylengths*. If *querymags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed (on the GPU) from the count matrix.

        Because of hardware limitations, the query matrices (*qsmilesmat* and *qcountsmat*) must have
        no more than 65,536 rows (molecules) and 32,768 columns (Lingos). Larger computations must be performed in tiles.
        """
        # Set up lingo and count matrices on device #{{{

        # padded_array will handle making matrix contiguous
        tempqlmat = self._padded_array(qsmilesmat.T)
        # TODO: use IMAGE2D_MAX_HEIGHT/WIDTH
        if tempqlmat.shape[1] > 65536 or tempqlmat.shape[0] > 32768:
            raise ValueError("Error: query matrix is not allowed to have more than 65536 rows (molecules) or 32768 columns (LINGOs) (both padded to multiple of 16). Dimensions = (%d,%d)"%tempqlmat.shape)
        self.gpu.qsmiles = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=tempqlmat)

        tempqcmat = self._padded_array(qcountsmat.T)
        self.gpu.qcounts = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=tempqcmat)
            
        #descriptor = cuda.ArrayDescriptor()
        #descriptor.width  = tempqcmat.shape[1]
        #descriptor.height = tempqcmat.shape[0]
        #descriptor.format = cuda.array_format.UNSIGNED_INT32
        #descriptor.num_channels = 1
        #self.gpu.tex2lq.set_address_2d(self.gpu.qsmiles,descriptor,tempqlmat.strides[0])
        #self.gpu.tex2cq.set_address_2d(self.gpu.qcounts,descriptor,tempqcmat.strides[0])
        #print "Set up query textures with stride=",tempqmat.strides[0]
        #self.gpu.stream.synchronize()
        self.qPitchTInInt = numpy.int32(tempqlmat.shape[1])
        del tempqlmat
        del tempqcmat
        #}}}

        self.qshape = qsmilesmat.shape
        self.nquery = qsmilesmat.shape[0]
        #print "Query shape=",self.qshape,", nquery=",self.nquery

        # Transfer query lengths array to GPU
        self.gpu.ql_gpu = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf = querylengths)

        # Allocate buffers for query set magnitudes
        self.gpu.qmag_gpu = cl.Buffer(self.gpu.context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf = querymags)
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

    def getTanimotoRow_async(self,rowidx): #{{{
        """Compute the single Tanimoto row *row* corresponding to comparing every SMILES string
        in the query set with the single reference SMILES string having index *row* in the reference set, and store it
        as the most recent asynchronous result.

        This method is asynchronous (it will return before the row has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #if self.gpu.rsmiles is None or self.gpu.qsmiles is None:
        #    raise 

        #self.resultmatrix = numpy.empty((1,self.nquery),dtype=numpy.float32)
        #self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)

        #self.gpu.singleRowKernel.prepare("iPPiii",block=(192,1,1),shared=int(2*4*self.rlengths[rowidx]),\
        #                                 texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cr,self.gpu.tex2cq])

        #self.gpu.singleRowKernel.prepared_async_call((30,1),self.gpu.stream,
        #         self.rlengths[rowidx],self.gpu.ql_gpu,self.gpu.gpumatrix,self.qshape[0],self.qshape[1],rowidx)

        #self.last_async_width = self.nquery
        #self.retrieveAsyncResult = self.retrieveAsyncMatrixResult
        #return
    #}}}

    def getMultipleRows(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in the reference set,
        and stores this block as the most recent asynchronous result.

        This method is synchronous (it will not return until the block has been completely computed).
        """
        if rowbase < 0 or rowlimit > self.nref:
            raise

        # Pad rows out to 64 byte pitch
        rowpitchInFloat = 16*((self.nquery+15)/16)

        # Using pagelocked memory and async copy seems to actually slow us down
        # on large tiled calculations
        self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInFloat),dtype=numpy.float32)
        self.gpu.gpumatrix = cl.Buffer(self.gpu.context,cl.mem_flags.WRITE_ONLY,size=self.resultmatrix.nbytes)

        # With precalculated magnitudes
        lmem_bytes = int(2*4*max(self.rlengths[rowbase:rowlimit]))
        threads_per_block = 192
        self.gpu.multiRowKernel(self.gpu.queue,(threads_per_block*(rowlimit-rowbase),),
                                               self.gpu.rsmiles,self.gpu.rcounts,self.gpu.rl_gpu,self.gpu.rmag_gpu,
                                               self.refPitchInInt,
                                               self.gpu.qsmiles,self.gpu.qcounts,self.gpu.ql_gpu,self.gpu.qmag_gpu,
                                               self.qPitchTInInt,
                                               self.gpu.gpumatrix, numpy.int32(rowpitchInFloat),
                                               numpy.int32(self.qshape[0]),numpy.int32(self.qshape[1]),numpy.int32(rowbase),
                                               cl.LocalMemory(lmem_bytes),cl.LocalMemory(lmem_bytes),
                                               local_size=(threads_per_block,))

        cl.enqueue_read_buffer(self.gpu.queue,self.gpu.gpumatrix,self.resultmatrix).wait()
        return self.resultmatrix[:,0:self.nquery]
    #}}}

    def getMultipleRows_async(self,rowbase,rowlimit): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in the reference set,
        and stores this block as the most recent asynchronous result.

        This method is asynchronous (it will return before the block has been completely computed). After
        calling this method, check :meth:`asyncOperationsDone`; once that method returns True, the result
        may be retrieved by calling :meth:`retrieveAsyncResult`.
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
    #}}}

    def getMultipleHistogrammedRows(self,rowbase,rowlimit): #{{{
        """TODO

        This method is synchronous (it will not return until the block has been completely computed).
        """
        self.getMultipleHistogrammedRows_async(rowbase,rowlimit)
        return self.retrieveAsyncResult()
    #}}}

    def getMultipleHistogrammedRows_async(self,rowbase,rowlimit): #{{{
        """TODO
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #if rowbase < 0 or rowlimit > self.nref:
        #    raise

        ## Pad rows out to 64 byte pitch - 112 is smallest multiple of 16 larger than the 101 needed
        #rowpitchInInt32 = 112

        ## Using pagelocked memory and async copy seems to actually slow us down
        ## on large tiled calculations
        ##if self.resultrows is not None and self.resultrows.nbytes != ((rowlimit-rowbase)*rowpitchInFloat*4):
        ##    print "Reallocating resultrows..."
        ##    del self.resultrows
        ##self.resultrows = cuda.pagelocked_empty((rowlimit-rowbase,rowpitchInFloat),dtype=numpy.float32)
        #self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInInt32),dtype=numpy.int32)
        #self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)

        ## With precalculated magnitudes
        #nthreads = 192
        ##shared=int(2*4*max(self.rlengths[rowbase:rowlimit]))+101*4*(nthreads/32)
        ##print "Asking for",shared,"bytes of shmem"
        #self.gpu.mrLingoHistKernel.prepare("PPPPPiiii",block=(nthreads,1,1),
        #                                   shared=int(2*4*max(self.rlengths[rowbase:rowlimit]))+101*4*(nthreads/32),\
        #                                   texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cq,self.gpu.tex2cr])

        #self.gpu.mrLingoHistKernel.prepared_async_call((rowlimit-rowbase,1),\
        #                                               self.gpu.stream,\
        #                                               self.gpu.rl_gpu, self.gpu.ql_gpu, self.gpu.rmag_gpu, self.gpu.qmag_gpu,\
        #                                               self.gpu.gpumatrix, rowpitchInInt32, self.qshape[0], self.qshape[1], rowbase)
        ##cuda.memcpy_dtoh_async(self.resultrows,self.gpu.gpurows,stream=self.gpu.stream)

        #self.last_async_width = 101
        #self.retrieveAsyncResult = self.retrieveAsyncMatrixResult

        #return
    #}}}

    def getNeighbors_async(self,rowbase,rowlimit,lowerbound,upperbound=1.1,maxneighbors=None): #{{{
        """todo
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #if rowbase < 0 or rowlimit > self.nref:
        #    raise
        #if maxneighbors is None:
        #    maxneighbors = self.nquery

        ## Pad rows out to 64 byte pitch
        #rowpitchInInt32 = 16*((maxneighbors+15)/16)

        #self.resultmatrix = numpy.empty((rowlimit-rowbase,rowpitchInInt32),dtype=numpy.int32)
        ##print "Allocated result matrix:",self.resultmatrix.shape
        #self.resultvector = numpy.empty((rowlimit-rowbase,),dtype=numpy.int32)
        ##print "Allocated result vector:",self.resultvector.shape
        #self.gpu.gpumatrix = cuda.mem_alloc(self.resultmatrix.nbytes)
        #self.gpu.gpuvector = cuda.mem_alloc(self.resultvector.nbytes)

        ## With precalculated magnitudes
        #self.gpu.nbrKernel.prepare("PPPPPiPffiii",block=(128,1,1),shared=int(2*4*max(self.rlengths[rowbase:rowlimit])),\
        #                           texrefs=[self.gpu.tex2lq,self.gpu.tex2lr,self.gpu.tex2cq,self.gpu.tex2cr])

        #self.gpu.nbrKernel.prepared_async_call((rowlimit-rowbase,1),\
        #                                       self.gpu.stream,\
        #                                       self.gpu.rl_gpu, self.gpu.ql_gpu, self.gpu.rmag_gpu, self.gpu.qmag_gpu,\
        #                                       self.gpu.gpumatrix, rowpitchInInt32, self.gpu.gpuvector,
        #                                       lowerbound,upperbound,
        #                                       self.qshape[0], self.qshape[1], rowbase)

        #self.last_async_width = maxneighbors
        #self.retrieveAsyncResult = self.retrieveAsyncMatrixVectorResult_trimmed
        #return 
    #}}}

    def getNeighbors(self,rowbase,rowlimit,lowerbound,upperbound=1.1,maxneighbors=None): #{{{
        """todo
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #self.getNeighbors_async(rowbase,rowlimit,lowerbound,upperbound,maxneighbors)
        #return self.retrieveAsyncResult()
    #}}}

    def retrieveAsyncMatrixResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #self.gpu.stream.synchronize()
        # 
        #cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        #self.gpu.gpumatrix.free()
        #
        #self.retrieveAsyncResult = self.retrieveAsyncResultBase
        #
        #return self.resultmatrix[:,0:self.last_async_width]
    #}}}

    def retrieveAsyncMatrixVectorResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        UPDATE THIS for neighbors:
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #self.gpu.stream.synchronize()
        #
        #print "Memcpy size: %d byte matrix, %d byte vector"%(self.resultmatrix.nbytes,self.resultvector.nbytes)
        #cuda.memcpy_dtoh(self.resultvector,self.gpu.gpuvector)
        #self.gpu.gpuvector.free()
        #t = time.time()
        #cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        #t2 = time.time()
        #print "Achieved %.2f MB/s bandwidth in %d x %d (%d byte) matrix copy"%((self.resultmatrix.nbytes/1048576.0)/(t2-t),self.resultmatrix.shape[0],self.resultmatrix.shape[1],self.resultmatrix.nbytes)
        #self.gpu.gpumatrix.free()
        #
        #self.retrieveAsyncResult = self.retrieveAsyncResultBase
        #
        #return (self.resultmatrix[:,0:self.last_async_width],self.resultvector)
    #}}}

    def retrieveAsyncMatrixVectorResult_trimmed(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`).

        Note that this result is only guaranteed to be valid if no operations have been run on this object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, or an asynchronous operation is still pending,
        (i.e., :meth:`asyncOperationsDone` is False), result is undefined.
        UPDATE THIS for neighbors:
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #self.gpu.stream.synchronize()
        #
        ##print "Memcpy size: %d byte matrix, %d byte vector"%(self.resultmatrix.nbytes,self.resultvector.nbytes)
        #cuda.memcpy_dtoh(self.resultvector,self.gpu.gpuvector)
        #self.gpu.gpuvector.free()

        ## We only need to retrieve as many columns as specified by the max of the vector elements
        #ncols = numpy.max(self.resultvector)

        #matrixcopy = cuda.Memcpy2D()
        #elttype = self.resultmatrix.dtype
        #matrixcopy.set_src_device(self.gpu.gpumatrix)
        #matrixcopy.src_pitch = self.resultmatrix.shape[1]*elttype.itemsize
        #self.resultmatrix = numpy.empty((self.resultmatrix.shape[0],ncols),dtype=elttype)
        #matrixcopy.set_dst_host(self.resultmatrix)
        #matrixcopy.dst_pitch=self.resultmatrix.shape[1]*elttype.itemsize
        #matrixcopy.width_in_bytes=self.resultmatrix.shape[1]*elttype.itemsize
        #matrixcopy.height=self.resultmatrix.shape[0]
        #
        ##cuda.memcpy_dtoh(self.resultmatrix,self.gpu.gpumatrix)
        #matrixcopy(True)
        #self.gpu.gpumatrix.free()

        #self.retrieveAsyncResult = self.retrieveAsyncResultBase

        ##return (self.resultmatrix[:,0:self.last_async_width],self.resultvector)
        #return (self.resultmatrix,self.resultvector)
    #}}}

    def retrieveAsyncResultBase(self): #{{{
        """TODO
        """
        raise NotImplementedException("Error: tried to call retrieveAsyncResult without a pending asynchronous result")
    #}}}
    def retrieveAsyncResult(self): #{{{
        """TODO
        """
        raise NotImplementedException("Error: tried to call retrieveAsyncResult without a pending asynchronous result")
    #}}}
    
    def asyncOperationsDone(self): #{{{
        """Return True if all asynchronous operations on this object have completed.
        """
        raise NotImplementedException("Asynchronous operations not currently implemented")
        #return self.gpu.stream.is_done()
    #}}}
