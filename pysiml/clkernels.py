#!/usr/bin/python

"""This module provides the user-facing API for calculating LINGO chemical
similarities using the SIML algorithm on CUDA-capable GPUs.
This module provides the OpenCL C source code for kernels used
by the OCLLingo module.
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



kernelsource = """
__kernel void accumulateRefMagnitudes(__global const int* devRLingoCounts,__global const int* devRLingoLengths,
                                      __global int* devRSetMagnitudes,const int pitchInInt)
{ //{{{
    const int molidx = get_group_id(0);
    __global const int* wgCounts = devRLingoCounts + molidx*pitchInInt;
    // TODO: we assume wg-size is 64
    local int shmem[64];
    const int localid = get_local_id(0);
    if (localid == 0)
      shmem[0] = devRLingoLengths[molidx];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int setLength = shmem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    shmem[localid] = 0;

    // Each work-item accumulates its own counter
    for (int base = 0; base < setLength; base += 64) {
        if ((base + localid) < setLength) {
            shmem[localid] += wgCounts[base+localid];
        } 
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Parallel reduce the sum
    for (int stride = 32; stride; stride >>= 1) {
        if (localid < stride) shmem[localid] += shmem[localid+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localid == 0) devRSetMagnitudes[molidx] = shmem[0];
    //if (localid == 0) devRSetMagnitudes[molidx] = devRLingoCounts[0+8];
    return;
} //}}}

float deviceMultisetTanimoto3_grgf_bT_sync_mags(__global const int* a,  __global const int* asize,  const int alen, const int amag,
                                                __global const int* bT, __global const int* bsizeT, const int blen, const int bmag,
                                                const int bmaxlen, const int bOffset,const int nquery,
                                                const int pitchTinInt) 
{ //{{{
    // Version of Tanimoto code that uses sparse vector interpretation
    // Use vector Tanimoto eqn with multiply -> min, add -> add
    //  (Induces L1 norm since all elements here are positive)
    //  <A,B> = |A \cap B|, <A,A>+<B,B>-<A,B> = |A \cup B|
    // Takes fit multisets in transposed representation. Syncs on rows to deal with lack of 2D texturing.
    int i=0,j=0;
    int isct=0;
    int bj,bsizej;

    // Special-case the empty set
    if (amag == 0 || bmag == 0) return 0.0f;

    while (j < bmaxlen) {
        if ((bOffset < nquery) && (j < blen)) {
            // Preloads here coalesce gmem access
            bj = *(bT + j*pitchTinInt + bOffset);
            bsizej = *(bsizeT + j*pitchTinInt + bOffset);
            while ( i < alen && a[i] < bj) {
                i++;
            }
            // a[i] >= bj or i == alen
            if (i < alen && a[i] == bj) {
                //bsizej = bsizeT + j*pitchTinInt + bOffset
                isct += min(asize[i],bsizej);
                i++;
            } 
            // a[i] > b[j] or i == alen
        }
        j++;
        // We don't have any memory writes to order
        barrier(0);
    }
    //return isct/((float)(amag+bmag-isct));
    float Union = amag+bmag-isct;
    //return __fdiv_rn(isct,Union);
    return isct/((float)Union);
} //}}}

float deviceMultisetTanimoto3_lrgf_bT_sync_mags(__local const int* a,  __local const int* asize,  const int alen, const int amag,
                                                __global const int* bT, __global const int* bsizeT, const int blen, const int bmag,
                                                const int bmaxlen, const int bOffset,const int nquery,
                                                const int pitchTinInt) 
{ //{{{
    // Version of Tanimoto code that uses sparse vector interpretation
    // Use vector Tanimoto eqn with multiply -> min, add -> add
    //  (Induces L1 norm since all elements here are positive)
    //  <A,B> = |A \cap B|, <A,A>+<B,B>-<A,B> = |A \cup B|
    // Takes fit multisets in transposed representation. Syncs on rows to deal with lack of 2D texturing.
    int i=0,j=0;
    int isct=0;
    int bj,bsizej;

    // Special-case the empty set
    if (amag == 0 || bmag == 0) return 0.0f;

    while (j < bmaxlen) {
        if ((bOffset < nquery) && (j < blen)) {
            // Preloads here coalesce gmem access
            bj = *(bT + j*pitchTinInt + bOffset);
            bsizej = *(bsizeT + j*pitchTinInt + bOffset);
            while ( i < alen && a[i] < bj) {
                i++;
            }
            // a[i] >= bj or i == alen
            if (i < alen && a[i] == bj) {
                //bsizej = bsizeT + j*pitchTinInt + bOffset
                isct += min(asize[i],bsizej);
                i++;
            } 
            // a[i] > b[j] or i == alen
        }
        j++;
        // We don't have any memory writes to order
        barrier(0);
    }
    //return isct/((float)(amag+bmag-isct));
    float Union = amag+bmag-isct;
    //return __fdiv_rn(isct,Union);
    return isct/((float)Union);
} //}}}

__kernel void gpuLingoSim3_multirow_grgf_T_sync(__global const int* grLingos,__global const int* grCounts,
                                                __global const int* grLengths,__global const int* grMags,
                                                const int refPitchInInt,
                                                __global const int* gqLingosT,__global const int* gqCountsT,
                                                __global const int* gqLengths,__global const int* gqMags,
                                                const int qPitchTInInt,
                                                __global float* gTanimotoRows, const int outPitchInFloat,
                                                const int nquery, const int maxlingos, const int refbase)
{ //{{{
    const int refidx = refbase + get_group_id(0);
    const __global int* wgRefLingos = grLingos + refidx*refPitchInInt;
    const __global int* wgRefCounts = grCounts + refidx*refPitchInInt;
    const int wgRefLength = grLengths[refidx];
    const int wgRefMag = grMags[refidx];
    // This is done wrt groupid, not refidx, because we might be outputting a tile smaller than refcount
    __global float* wgTanimotoRow = gTanimotoRows + get_group_id(0)*outPitchInFloat;

    int myOffset;
    for (int base = 0; base < nquery; base += get_local_size(0)) {
        myOffset = base + get_local_id(0);
        float tanimoto;
        tanimoto = deviceMultisetTanimoto3_grgf_bT_sync_mags(wgRefLingos,wgRefCounts,wgRefLength,wgRefMag,
                                                             gqLingosT, gqCountsT, gqLengths[myOffset],gqMags[myOffset],
                                                             maxlingos, myOffset, nquery,
                                                             qPitchTInInt);
        if (myOffset < nquery) {
            wgTanimotoRow[myOffset] = tanimoto;
        }
    }
    return;
} //}}}


__kernel void gpuLingoSim3_multirow_lrgf_T_sync(__global const int* grLingos,__global const int* grCounts,
                                                __global const int* grLengths,__global const int* grMags,
                                                const int refPitchInInt,
                                                __global const int* gqLingosT,__global const int* gqCountsT,
                                                __global const int* gqLengths,__global const int* gqMags,
                                                const int qPitchTInInt,
                                                __global float* gTanimotoRows, const int outPitchInFloat,
                                                const int nquery, const int maxlingos, const int refbase,
                                                __local int* lrLingos, __local int* lrCounts)
{ //{{{
    const int refidx = refbase + get_group_id(0);
    //const __global int* wgRefLingos = grLingos + refidx*refPitchInInt;
    //const __global int* wgRefCounts = grCounts + refidx*refPitchInInt;
    //const int wgRefLength = grLengths[refidx];
    //const int wgRefMag = grMags[refidx];
    int wgRefLength, wgRefMag;
    //if (get_local_id(0) == 0) {
        wgRefLength = grLengths[refidx];
        wgRefMag = grMags[refidx];
    //}
    //barrier(CLK_LOCAL_MEM_FENCE);
    // Preload reference lingos/counts
    int myOffset;
    for (int base = 0; base < wgRefLength; base += get_local_size(0)) {
        myOffset = base + get_local_id(0);
        if (myOffset < wgRefLength) {
            lrLingos[myOffset] = *(grLingos + refidx*refPitchInInt + myOffset);    
            lrCounts[myOffset] = *(grCounts + refidx*refPitchInInt + myOffset);    
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // This is done wrt groupid, not refidx, because we might be outputting a tile smaller than refcount
    __global float* wgTanimotoRow = gTanimotoRows + get_group_id(0)*outPitchInFloat;

    for (int base = 0; base < nquery; base += get_local_size(0)) {
        myOffset = base + get_local_id(0);
        float tanimoto;
        tanimoto = deviceMultisetTanimoto3_lrgf_bT_sync_mags(lrLingos,lrCounts,wgRefLength,wgRefMag,
                                                             gqLingosT,gqCountsT,gqLengths[myOffset],gqMags[myOffset],
                                                             maxlingos, myOffset,nquery,
                                                             qPitchTInInt);
        if (myOffset < nquery) {
            wgTanimotoRow[myOffset] = tanimoto;
        }
    }
    return;
} //}}}

"""

