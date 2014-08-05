#!/usr/bin/python

"""This module provides the user-facing API for calculating LINGO chemical
similarities using the SIML algorithm on CPUs.
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


import _CPULingo
import numpy

class CPULingo:
    """Object to handle computation of LINGO similarities on the CPU
    """
    def __init__(self): #{{{
        """Initializes a CPULingo comparator object. Requires no parameters.
        """
        self.rlingos = None
        self.rcounts = None
        self.rlengths = None
        self.rmags = None
        self.qlingos = None
        self.qcounts = None
        self.qlengths = None
        self.qmags = None
        self.lastasyncresult = None
        self.needToWarnParallel = not _CPULingo.supportsParallel()
    #}}}

    def supportsParallel(self): #{{{
        """Return True if this installation of pySIML was built with OpenMP support for parallel calculations.

        Note that even if this function returns False, the :meth:`getMultipleRows` and :meth:`getMultipleRows_async`
        methods can still be called with nprocs > 1, but only one processor will actually be used
        """
        return _CPULingo.supportsParallel()
    #}}}

    def set_refsmiles(self,refsmilesmat,refcountsmat,reflengths,refmags=None): #{{{
        """Sets the reference SMILES set to use Lingo matrix *refsmilesmat*, count matrix *refcountsmat*,
        and length vector *reflengths*. If *refmags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed from the count matrix.
        """
        # TODO: Should verify type, shape, etc here
        self.rlingos = refsmilesmat
        self.rcounts = refcountsmat
        self.rlengths = reflengths
        if refmags is not None:
            self.rmags = refmags
        else:
            self.rmags = numpy.int32(reflengths.sum(1))
        return
    #}}}

    def set_qsmiles(self,qsmilesmat,qcountsmat,querylengths,querymags=None): #{{{
        """Sets the query SMILES set to use Lingo matrix *qsmilesmat*, count matrix *qcountsmat*,
        and length vector *querylengths*. If *querymags* is provided, it will be used as the magnitude
        vector; else, the magnitude vector will be computed from the count matrix.
        """
        # Should verify type, shape, etc here
        self.qlingos = qsmilesmat
        self.qcounts = qcountsmat
        self.qlengths = querylengths
        if querymags is not None:
            self.qmags = querymags
        else:
            self.qmags = numpy.int32(querylengths.sum(1))

        return
    #}}}

    def _checkInitialized(self): #{{{
        """Returns True if both the reference and query SMILES sets have been initialized and False otherwise.
        """
        if self.rlingos is None or self.rcounts is None or self.rlengths is None or self.rmags is None or\
           self.qlingos is None or self.qcounts is None or self.qlengths is None or self.qmags is None:
           raise ValueError("CPULingo object not fully initialized")
    #}}}

    def getMultipleRows_async(self,rowbase,rowlimit,nprocs=1): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in the reference set,
        and stores this block of rows internally as the last asynchronous result value.

        If pySIML has been built with OpenMP enabled, *nprocs* may be set higher than 1 to parallelize computations
        over multiple CPUs (each CPU will handle a disjoint set of rows). If called with *nprocs* larger than 1 on a
        non-OpenMP build of pySIML, print a warning to stderr and compute with one CPU.

        To retrieve the result block, call :meth:`retrieveAsyncResult`.

        Note that this function is actually synchronous, due to the limitations of running on the CPU;
        it will not return until the block has been completely calculated.
        """
        # We can't actually do an asynchronous calculation, so fake it
        self._checkInitialized()
        if nprocs <= 1:
            self.lastasyncresult = _CPULingo.getTanimotoBlock(self.rlingos[rowbase:rowlimit,:],self.rcounts[rowbase:rowlimit,:],
                                                              self.rmags[rowbase:rowlimit],self.rlengths[rowbase:rowlimit],
                                                              self.qlingos,self.qcounts,self.qmags,self.qlengths)
        else:
            if self.needToWarnParallel:
                print "Warning: called getMultipleRows_async requesting more than one CPU and pysiml not built with OpenMP support. Only using one CPU."
                self.needToWarnParallel = False
            self.lastasyncresult = _CPULingo.getTanimotoBlockParallel(
                                                self.rlingos[rowbase:rowlimit,:],self.rcounts[rowbase:rowlimit,:],
                                                self.rmags[rowbase:rowlimit],self.rlengths[rowbase:rowlimit],
                                                self.qlingos,self.qcounts,self.qmags,self.qlengths,nprocs)

        return
    #}}}

    def getTanimotoRow_async(self,row): #{{{
        """Computes the single Tanimoto row *row* corresponding to comparing every SMILES string
        in the query set with the single reference SMILES string having index *row* in the reference set,
        and stores this row internally as the last asynchronous result value.

        To retrieve the result row, call :meth:`retrieveAsyncResult`.

        Note that this function is actually synchronous, due to the limitations of running on the CPU;
        it will not return until the row has been completely calculated.
        """
        self.getMultipleRows_async(row,row+1)
        return
    #}}}

    def getTanimotoRow(self,row): #{{{
        """Returns the single Tanimoto row *row* corresponding to comparing every SMILES string
        in the query set with the single reference SMILES string having index *row* in the reference set.
        """
        self.getTanimotoRow_async(row)
        return self.retrieveAsyncResult()
    #}}}

    def getMultipleRows(self,rowbase,rowlimit,nprocs=1): #{{{
        """Computes multiple Tanimoto rows *rowbase:rowlimit* corresponding to comparing every SMILES string
        in the query set with the reference SMILES strings having index *row*, *row+1*, ..., *rowlimit-1* in the reference set,
        and returns this block of rows.

        If pySIML has been built with OpenMP enabled, *nprocs* may be set higher than 1 to parallelize computations
        over multiple CPUs (each CPU will handle a disjoint set of rows). If called with *nprocs* larger than 1 on a
        non-OpenMP build of pySIML, print a warning to stderr and compute with one CPU.
        """
        if nprocs > 1 and self.needToWarnParallel:
            print "Warning: called getMultipleRows requesting more than one CPU and pysiml not built with OpenMP support. Only using one CPU."
            self.needToWarnParallel = False
        self.getMultipleRows_async(rowbase,rowlimit,nprocs)
        return self.retrieveAsyncResult()
    #}}}

    def retrieveAsyncResult(self): #{{{
        """Returns result from last asynchronous computation (:meth:`getTanimotoRow_async` or :meth:`getMultipleRows_async`). 
        
        Note that this result is only guaranteed to be valid if no operations have been run on this CPULingo object since the
        asynchronous call, except for :meth:`asyncOperationsDone` and :meth:`retrieveAsyncResult`.

        If no asynchronous operations have been invoked on this object, result is undefined.
        """
        return self.lastasyncresult
    #}}}

    def asyncOperationsDone(self):
        """Return True if all asynchronous operations on this object have completed.

        In current implementation, always returns True.
        """
        return True
