#!/usr/bin/python

"""This module provides "compilers" to convert SMILES strings into the
sparse-vector representation required for SIML. C and pure-Python 
implementations are provided.
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
import struct
import string

try:
    import _ccompiler
    def _getSMILESname(smi):
        fields = smi.split()
        if len(fields) > 1:
            return fields[1]
        else:
            return ""
    def cSMILEStoMatrices(smileslist):
        """Convert the sequence of SMILES strings *smileslist* into a SIML SMILES set
        and list of molecule names (if present in the SMILES strings). Uses the SIML
        compiler C extension. See pySIML preprocessing documentation for details on
        transformations performed on the SMILES strings. Note that this does NOT perform
        the same transformations as the pure-Python version :meth:`SMILEStoMatrices`.

        Returns a tuple of 5 values: a Lingo matrix, a count matrix, a length vector, 
        a magnitude vector, and a list of molecule names (all but the molecule names
        make up the "SMILES set").
        """
        # This method takes two passes through the list - one to generate
        # the matrices, and one to grab the names
        # In case we're given a file (which can't be iterated through again)
        # store the offset so we can rewind it for the second pass
        offset = None
        try:
            offset = smileslist.tell()
        except:
            pass

        # Use the C extension to calculate matrices
        (lingos,counts,lengths,mags) = _ccompiler.SMILEStoMatrices(smileslist)

        # Grab names in Python
        if offset is not None:
            smileslist.seek(offset)
        names = map(_getSMILESname,smileslist)

        return (lingos,counts,lengths,mags,names)
except:
    # If we can't get the C module, still define the function
    def cSMILEStoMatrices(smileslist):
        """Convert the sequence of SMILES strings *smileslist* into a SIML SMILES set
        and list of molecule names (if present in the SMILES strings). Uses the SIML
        compiler C extension. See pySIML preprocessing documentation for details on
        transformations performed on the SMILES strings. Note that this does NOT perform
        the same transformations as the pure-Python version :meth:`SMILEStoMatrices`.

        Returns a tuple of 5 values: a Lingo matrix, a count matrix, a length vector, 
        a magnitude vector, and a list of molecule names (all but the molecule names
        make up the "SMILES set").
        """
        raise NotImplementedError("Unable to import _ccompiler extension module; this method is unavailable")

def SMILEStoMultiset(smiles): #{{{
    """Returns Lingo and count vectors for a single SMILES string *smiles*, as would
    correspond to a row in the Lingo or count matrices from :meth:`cSMILEStoMatrices` or
    :meth:`SMILEStoMatrices`. Performs no transformations on *smiles* prior to conversion.

    Note that in general, the results of this function will not be the same as
    those obtained from :meth:`SMILEStoMatrices` or :meth:`cSMILEStoMatrices` because
    this function does not preprocess the input strings.
    """
    def addIntsToBag(ns,lingo_to_count):
        for i in xrange(ns.shape[0]):
            lingo = ns[i]
            if lingo not in lingo_to_count:
                lingo_to_count[lingo] = 1
            else:
                lingo_to_count[lingo] += 1
    
    # Handle each shift separately
    #print "Using shift crop method"
    modlength = len(smiles) % 4
    lingo_to_count = {}
    for shift in range(4):
        tailcrop = -((modlength + (4-shift)) % 4)
        if tailcrop == 0:
            ns = numpy.fromstring(smiles[shift:],dtype=numpy.int32)
        else:
            ns = numpy.fromstring(smiles[shift:tailcrop],dtype=numpy.int32)
        addIntsToBag(ns,lingo_to_count)
    counts = numpy.empty((len(lingo_to_count),1),dtype=numpy.int32)
    lingos = numpy.empty_like(counts)
    for i,lingo in enumerate(sorted(lingo_to_count.keys())):
        lingos[i] = lingo
        counts[i] = lingo_to_count[lingo]

    return (lingos,counts)
#}}}

def preprocessNumbers(smi,xtable=None):
    """Given a SMILES string, return a copy of the string with the same translations
    performed on it as would be done by the pure-Python preprocessor
    :meth:`SMILEStoMatrices`. 
    
    This method is primarily useful to compare the results
    of SIML Tanimoto calculation functions with those from other LINGO calculation
    packages, to ensure that identical SMILES strings are given to each.
    
    *xtable* is an internal parameter and should always be set to None when called
    from user code.
    """
    if xtable is None:
        xtable = string.maketrans('123456789','000000000')
    return string.translate(smi,xtable)

def SMILEStoMatrices(smileslist):
    """Convert the sequence of SMILES strings *smileslist* into a SIML SMILES set
    and list of molecule names (if present in the SMILES strings). Uses a pure
    Python implementation. See pySIML preprocessing documentation for details on
    transformations performed on the SMILES strings. Note that this does NOT perform
    the same transformations as the C version :meth:`cSMILEStoMatrices`.

    Returns a tuple of 5 values: a Lingo matrix, a count matrix, a length vector, 
    a magnitude vector, and a list of molecule names (all but the molecule names
    make up the "SMILES set").
    """
    # Translation table to map all digits to zero
    # Note that this is not strictly correct - it will transform things like [NH2+] to [NH0+]
    xtable = string.maketrans('123456789','000000000')

    lingos = []
    counts = []
    names = []
    maxw = 0
    for line in smileslist:
        fields = line.split()
        #smiles = string.translate(fields[0],xtable)
        smiles = preprocessNumbers(fields[0],xtable)
        l,c = SMILEStoMultiset(smiles)
        lingos.append(l)
        counts.append(c)
        if len(fields) > 1:
            names.append(fields[1])
        else:
            names.append("")
        maxw = max(maxw,l.shape[0])

    lingomatrix = numpy.zeros( (len(lingos),maxw) ,dtype=numpy.int32)
    countmatrix = numpy.zeros_like(lingomatrix)
    lingolengths = numpy.empty((len(lingos),1),dtype=numpy.int32)

    for i in xrange(len(lingos)):
        l = lingos[i].shape[0]
        lingolengths[i] = l
        lingomatrix[i,0:l] = lingos[i][:,0]
        countmatrix[i,0:l] = counts[i][:,0]
    lingomags = numpy.int32(countmatrix.sum(1))

    return (lingomatrix,countmatrix,lingolengths,lingomags,names)
