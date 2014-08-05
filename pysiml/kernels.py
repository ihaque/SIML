#!/usr/bin/python
"""This module provides the CUDA source code for GPU kernels used
by the GPULingo module.
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
texture<int32_t,2> tex2FitLingoMatrixT;
texture<int32_t,2> tex2FitLingoCountMatrixT;
texture<int32_t,2> tex2RefLingoMatrix;
texture<int32_t,2> tex2RefLingoCountMatrix;

__global__ void accumulateRefMagnitudes(int32_t* devRSetMagnitudes,int32_t* devRLingoLengths,int32_t nRefMols)
{ //{{{
    // Buffer magnitudes into shared memory to coalesce global writes
    // Use 2D texture to coalesce global reads
    extern __shared__ int32_t thdMagnitude[];
    for (int base = 0; base < nRefMols; base += gridDim.x * blockDim.x) {
        int molidx = base + blockDim.x*blockIdx.x + threadIdx.x;
        if (molidx < nRefMols) {
            thdMagnitude[threadIdx.x] = 0;
            for (int i = 0; i < devRLingoLengths[molidx]; i++) {
                thdMagnitude[threadIdx.x] += tex2D(tex2RefLingoCountMatrix,i,molidx);
            }
        }
        __syncthreads();
        if (molidx < nRefMols) devRSetMagnitudes[molidx] = thdMagnitude[threadIdx.x];
    }
} //}}}

__global__ void accumulateQueryMagnitudes(int32_t* devQSetMagnitudes,int32_t* devQLingoLengths,int32_t nQueryMols)
{ //{{{
    // Buffer magnitudes into shared memory to coalesce global writes
    // Use 2D texture to coalesce global reads
    extern __shared__ int32_t thdMagnitude[];
    for (int base = 0; base < nQueryMols; base += gridDim.x * blockDim.x) {
        int molidx = base + blockDim.x*blockIdx.x + threadIdx.x;
        if (molidx < nQueryMols) {
            thdMagnitude[threadIdx.x] = 0;
            for (int i = 0; i < devQLingoLengths[molidx]; i++) {
                thdMagnitude[threadIdx.x] += tex2D(tex2FitLingoCountMatrixT,molidx,i);
            }
        }
        __syncthreads();
        if (molidx < nQueryMols) devQSetMagnitudes[molidx] = thdMagnitude[threadIdx.x];
    }
} //}}}

__device__ float deviceMultisetTanimoto1_bT_tex2(int32_t* a,int32_t* asize,int32_t alen,int32_t blen,int32_t bmaxlen,int32_t bOffset) { //{{{
    // Version of Tanimoto code that uses explicit branches
    // Takes fit multisets in transposed representation and syncs on those.
    int32_t i=0,j=0;
    int32_t un=0,isct=0;

    int32_t bj,bsizej;

    // Special-case the empty set
    if (alen == 0 || blen == 0) return 0.0f;

    while (j < bmaxlen) {
        if (j < blen) {
            // Preloads here coalesce gmem access
            bj = tex2D(tex2FitLingoMatrixT,bOffset,j);
            bsizej = tex2D(tex2FitLingoCountMatrixT,bOffset,j);
            while ( i < alen && a[i] < bj) {
                un += asize[i];
                i++;
            }
            // a[i] >= bj or i == alen
            if (i < alen && a[i] == bj) {
                un += max(asize[i],bsizej);
                isct += min(asize[i],bsizej);
                i++;
            } else {
                un += bsizej;
            }
            // a[i] > b[j] or i == alen
        }
        j++;
    }
    while ( i < alen) {
        un += asize[i];
        i++;
    }
    float t1 = isct;
    float t2 = un;
    return __fdiv_rn(t1,t2);
    //return ((float)isct)/un;
} //}}}
__device__ float deviceMultisetTanimoto3_bT_tex2(int32_t* a,int32_t* asize,int32_t alen,const int32_t amag,
                                                 int32_t blen,int32_t bmaxlen,int32_t bOffset) { 
    //{{{
    // Version of Tanimoto code that uses sparse vector interpretation
    // Use vector Tanimoto eqn with multiply -> min, add -> add
    //  (Induces L1 norm since all elements here are positive)
    //  <A,B> = |A \cap B|, <A,A>+<B,B>-<A,B> = |A \cup B|
    // Takes fit multisets in transposed representation.
    int32_t i=0,j=0;
    int32_t bmag=0,isct=0;

    int32_t bj,bsizej;

    // Special-case the empty set
    if (alen == 0 || blen == 0) return 0.0f;

    while (j < bmaxlen) {
        if (j < blen) {
            // Preloads here coalesce gmem access
            bj = tex2D(tex2FitLingoMatrixT,bOffset,j);
            bsizej = tex2D(tex2FitLingoCountMatrixT,bOffset,j);
            bmag += bsizej;
            while ( i < alen && a[i] < bj) {
                i++;
            }
            // a[i] >= bj or i == alen
            if (i < alen && a[i] == bj) {
                isct += min(asize[i],bsizej);
                i++;
            } 
            // a[i] > b[j] or i == alen
        }
        j++;
    }
    //return isct/((float)(amag+bmag-isct));
    float tmp = amag+bmag-isct;
    return __fdiv_rn(isct,tmp);
}
//}}}

__device__ float deviceMultisetTanimoto3_bT_tex2_mags(int32_t* a,int32_t* asize,int32_t alen,const int32_t amag,const int32_t bmag,
                                                 int32_t blen,int32_t bmaxlen,int32_t bOffset) { 
    //{{{
    // Version of Tanimoto code that uses sparse vector interpretation
    // Use vector Tanimoto eqn with multiply -> min, add -> add
    //  (Induces L1 norm since all elements here are positive)
    //  <A,B> = |A \cap B|, <A,A>+<B,B>-<A,B> = |A \cup B|
    // Takes fit multisets in transposed representation.
    int32_t i=0,j=0;
    int32_t isct=0;
    int32_t bj,bsizej;

    // Special-case the empty set
    if (amag == 0 || bmag == 0) return 0.0f;

    while (j < bmaxlen) {
        if (j < blen) {
            // Preloads here coalesce gmem access
            bj = tex2D(tex2FitLingoMatrixT,bOffset,j);
            // It's actually faster to texture this every time, rather than conditionally. Weird.
            bsizej = tex2D(tex2FitLingoCountMatrixT,bOffset,j);
            while ( i < alen && a[i] < bj) {
                i++;
            }
            // a[i] >= bj or i == alen
            if (i < alen && a[i] == bj) {
                //bsizej = tex2D(tex2FitLingoCountMatrixT,bOffset,j);
                isct += min(asize[i],bsizej);
                i++;
            } 
            // a[i] > b[j] or i == alen
        }
        j++;
    }
    //return isct/((float)(amag+bmag-isct));
    float Union = amag+bmag-isct;
    return __fdiv_rn(isct,Union);
}
//}}}

__global__ void gpuLingoSim1_srgf_T_tex2ns(int32_t refLingoLength,int32_t* devQLingoLengths,float* tanimotos,int32_t nmols,int32_t maxlingos,int32_t refidx) { //{{{
    // Compares all molecules 0:N-1 against molecule refidx
    // Calculate nBlocks*nThreads tanimotos at a time
    // Places reference multiset into shared memory
    extern __shared__ int32_t shmem[];
    //int32_t refLingoLength = devLingoLengths[refidx];
    // Broadcast the reference multiset length from thread 0 through shared memory rather than using uncoalesced read
    //if (threadIdx.x == 0) shmem[0] = devLingoLengths[refidx];
    //__syncthreads();
    //int32_t refLingoLength = shmem[0];

    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            //refLingos[base+threadIdx.x] = *(devLingoMatrix + (refidx*pitchInInt32) + base+threadIdx.x);
            //refLingoCounts[base+threadIdx.x] = *(devLingoCountMatrix + (refidx*pitchInInt32) + base+threadIdx.x);
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();

    int32_t myOffset;
    for (int32_t base = 0; base < nmols; base+= gridDim.x * blockDim.x) {
        myOffset = base + blockDim.x*blockIdx.x + threadIdx.x;
        if (myOffset < nmols) {
            tanimotos[myOffset] = deviceMultisetTanimoto1_bT_tex2(refLingos,refLingoCounts,refLingoLength,devQLingoLengths[myOffset],maxlingos,myOffset);
        }
    }
} //}}}

__global__ void gpuLingoSim3_srgf_T_tex2ns(int32_t refLingoLength,int32_t* devQLingoLengths,float* tanimotos,int32_t nmols,int32_t maxlingos,int32_t refidx) 
{ //{{{
    // Compares all molecules 0:N-1 against molecule refidx
    // Calculate nBlocks*nThreads tanimotos at a time
    // Places reference multiset into shared memory
    extern __shared__ int32_t shmem[];
    //int32_t refLingoLength = devLingoLengths[refidx];
    // Broadcast the reference multiset length from thread 0 through shared memory rather than using uncoalesced read
    //if (threadIdx.x == 0) shmem[0] = devLingoLengths[refidx];
    //__syncthreads();
    //int32_t refLingoLength = shmem[0];

    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            //refLingos[base+threadIdx.x] = *(devLingoMatrix + (refidx*pitchInInt32) + base+threadIdx.x);
            //refLingoCounts[base+threadIdx.x] = *(devLingoCountMatrix + (refidx*pitchInInt32) + base+threadIdx.x);
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();
    // Let's be very stupid about this reduction for current expediency
    int32_t amag = 0;
    for (int i = 0; i < refLingoLength; i++) {
        amag += refLingoCounts[i];
    }

    int32_t myOffset;
    for (int32_t base = 0; base < nmols; base+= gridDim.x * blockDim.x) {
        myOffset = base + blockDim.x*blockIdx.x + threadIdx.x;
        if (myOffset < nmols) {
            tanimotos[myOffset] = deviceMultisetTanimoto3_bT_tex2(refLingos,refLingoCounts,refLingoLength,amag,devQLingoLengths[myOffset],maxlingos,myOffset);
        }
    }
}
//}}}

__global__ void gpuLingoSim1_multirow(int32_t* devRLingoLengths,int32_t* devQLingoLengths,
                                      float* tanimotoRows,int32_t pitchInFloat,
                                      int32_t nmols,int32_t maxlingos,int32_t refbase) 
{ //{{{
    // Compares all molecules 0:nmols against molecules refbase:refbase+gridDim.x
    // Calculate nBlocks*nThreads tanimotos at a time
    // Places reference multiset into shared memory
    extern __shared__ int32_t shmem[];
    const int32_t refidx = refbase + blockIdx.x;
    if (threadIdx.x == 0) {
        shmem[0] = devRLingoLengths[refidx];
    }
    __syncthreads();
    const int32_t refLingoLength = shmem[0];
    __syncthreads();

    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();
    // Let's be very stupid about this reduction for current expediency
    int32_t refSetMagnitude = 0;
    for (int i = 0; i < refLingoLength; i++) {
        refSetMagnitude += refLingoCounts[i];
    }

    int32_t myOffset;
    float* tanimotos = tanimotoRows + blockIdx.x*pitchInFloat;
    for (int32_t base = 0; base < nmols; base += blockDim.x) {
        myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            tanimotos[myOffset] = deviceMultisetTanimoto1_bT_tex2(refLingos,refLingoCounts,refLingoLength,devQLingoLengths[myOffset],maxlingos,myOffset);
        }
    }
}
//}}}

__global__ void gpuLingoSim3_multirow(int32_t* devRLingoLengths,int32_t* devQLingoLengths,
                                      int32_t* devRSetMagnitudes,int32_t* devQSetMagnitudes,
                                      float* tanimotoRows,int32_t pitchInFloat,
                                      int32_t nmols,int32_t maxlingos,int32_t refbase) 
{ //{{{
    // Compares all molecules 0:nmols against molecules refbase:refbase+gridDim.x
    // Calculate nBlocks*nThreads tanimotos at a time
    // Places reference multiset into shared memory
    extern __shared__ int32_t shmem[];
    const int32_t refidx = refbase + blockIdx.x;
    if (threadIdx.x == 0) {
        shmem[0] = devRLingoLengths[refidx];
        shmem[1] = devRSetMagnitudes[refidx];
    }
    __syncthreads();
    const int32_t refLingoLength = shmem[0];
    const int32_t refSetMagnitude = shmem[1];
    __syncthreads();

    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();

    int32_t myOffset;
    float* tanimotos = tanimotoRows + blockIdx.x*pitchInFloat;
    for (int32_t base = 0; base < nmols; base += blockDim.x) {
        myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            tanimotos[myOffset] = deviceMultisetTanimoto3_bT_tex2_mags(refLingos,refLingoCounts,refLingoLength,refSetMagnitude,devQSetMagnitudes[myOffset],devQLingoLengths[myOffset],maxlingos,myOffset);
        }
    }
}
//}}}

// These two functions modified from public domain licensed code in SimTK/OpenMM for stream compaction
__device__ int exclusivePrescan128(const int32_t* in, int32_t* outAndTemp) 
{ //{{{
    // Exclusive prefix scan over 128 elements
    // Assumes 128 threads
    // Taken from cuda SDK "scan" sample for naive scan, with small modifications
    const int n=128;
    int32_t* temp = outAndTemp;
    int pout = 1, pin = 0;

    // load input into temp
    // This is exclusive scan, so shift right by one and set first elt to 0
    temp[pout*n + threadIdx.x] = (threadIdx.x > 0) ? in[threadIdx.x-1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        __syncthreads();
        temp[pout*n+threadIdx.x] = temp[pin*n+threadIdx.x];
        if (threadIdx.x >= offset)
            temp[pout*n+threadIdx.x] += temp[pin*n+threadIdx.x - offset];
    }
    __syncthreads();
    return outAndTemp[127]+in[127]; // Return sum of all elements
} //}}}

__device__ int compactSIMDPrefixSum(const int32_t dataOffset,const int32_t* dsValid,int32_t* dsCompact) 
{ //{{{
    __shared__ int32_t dsLocalIndex[256];
    int numValid = exclusivePrescan128(dsValid,dsLocalIndex);
    //if (dsValid[threadIdx.x]) dsCompact[dsLocalIndex[threadIdx.x]] = dsData[threadIdx.x];
    if (dsValid[threadIdx.x]) dsCompact[dsLocalIndex[threadIdx.x]] = dataOffset+threadIdx.x;
    return numValid;
} //}}}


__global__ void gpuGetNeighbors_multirow(int32_t* devRLingoLengths,int32_t* devQLingoLengths,
                                         int32_t* devRSetMagnitudes,int32_t* devQSetMagnitudes,
                                         int32_t* neighborRows,int32_t pitchInInt32,int32_t* neighborCounts,
                                         const float neighborThreshold, const float upperBound,
                                         const int32_t nmols,const int32_t maxlingos,const int32_t refbase) 
{ //{{{
    // Compares all molecules 0:nmols against molecules refbase:refbase+gridDim.x
    // Calculate nBlocks*nThreads tanimotos at a time
    // Places reference multiset into shared memory
    // CALL ONLY WITH 128 THREADS/BLOCK!
    extern __shared__ int32_t shmem[];

    // We don't need a data block for stream compaction since the data is always threadIdx + an offset
    __shared__ int32_t isNeighbor[128],compactBlock[128];
    int rowNeighborCount = 0;

    const int32_t refidx = refbase + blockIdx.x;
    if (threadIdx.x == 0) {
        shmem[0] = devRLingoLengths[refidx];
        shmem[1] = devRSetMagnitudes[refidx];
    }
    __syncthreads();
    const int32_t refLingoLength = shmem[0];
    const int32_t refSetMagnitude = shmem[1];
    __syncthreads();

    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();

    int32_t myOffset;
    int32_t* neighbors = neighborRows + blockIdx.x*pitchInInt32;
    float myTanimoto;
    
    for (int32_t base = 0; base < nmols; base += blockDim.x) {
        // Compute neighbor index and 'neighborness' flags in waves of blockDim.x
        myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            int32_t qSetMagnitude = devQSetMagnitudes[myOffset];
            // We can avoid some full Tanimoto computations by a simple set bound on the Tanimoto:
            // T(A,B) = |A ^ B| / |A u B| <= min(|A|,|B|)/max(|A|,|B|)
            float tanimotoBound = min(refSetMagnitude,qSetMagnitude)/((float)(max(refSetMagnitude,qSetMagnitude)));
            isNeighbor[threadIdx.x] = int(tanimotoBound >= neighborThreshold);

            // Only if the upper bound is >= threshold bother calculating the exact Tanimoto
            if (isNeighbor[threadIdx.x]) {
                myTanimoto = deviceMultisetTanimoto3_bT_tex2_mags(refLingos,refLingoCounts,refLingoLength,\
                                                                  refSetMagnitude,qSetMagnitude,devQLingoLengths[myOffset],\
                                                                  maxlingos,myOffset);
                isNeighbor[threadIdx.x] = myTanimoto >= neighborThreshold && myTanimoto < upperBound;
            }
        } else {
            isNeighbor[threadIdx.x] = 0;
        }
        
        __syncthreads();

        // Compact the local list of neighbors
        int numValidBlock = compactSIMDPrefixSum(base,isNeighbor,compactBlock);
        __syncthreads();

        // Write as many neighbors out to the output array as we can
        //    - if our compacted element is valid, and it would not overflow the output buffer
        if (threadIdx.x < numValidBlock && (rowNeighborCount + threadIdx.x) < pitchInInt32)
            neighbors[rowNeighborCount + threadIdx.x] = compactBlock[threadIdx.x];
        
        // Increment the base address for the next wave of Tanimotos
        rowNeighborCount += numValidBlock;

        // If we've overflowed the neighbor list, terminate
        if (rowNeighborCount > pitchInInt32) break;
    }
    __syncthreads();
    // Neighbors have already been written; now write out the length of each neighbor list
    // (which will be > pitch if the list overflowed)
    if (threadIdx.x == 0) neighborCounts[blockIdx.x] = rowNeighborCount;
}
//}}}

"""

sm11_histogramming_kernels = """
__global__ void histogramMultipleRows101(float* tanimotoRows,int32_t pitchInFloat,int32_t nmols,int32_t* histogramRows,int32_t pitchInInt32) 
{ //{{{
    // 101 histogram bins (0, 0.01, ... 1.00)
    // sm11 version uses atomic adds to global memory
    const int histogramBins = 101;

    
    // Each block processes exactly one row
    const float* tanimotos = tanimotoRows + blockIdx.x*pitchInFloat;
    int32_t* glHistogram = histogramRows + blockIdx.x*pitchInInt32;

    // Initialize histogram
    if (threadIdx.x < histogramBins) glHistogram[threadIdx.x] = 0;
    __threadfence_block();
    __syncthreads();

    // Iterate over elements in the row
    for (int base = 0; base < nmols; base += blockDim.x) {
        int myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            float data = tanimotos[myOffset];
            unsigned index = (unsigned int)(data*100);
            atomicAdd(glHistogram+index,1);
        }
    }

    return;

} //}}}

__global__ void gpuHistogrammedLingo_multirow(int32_t* devRLingoLengths,int32_t* devQLingoLengths,
                                              int32_t* devRSetMagnitudes,int32_t* devQSetMagnitudes,
                                              int32_t* histogramRows,int32_t pitchInInt32,
                                              int32_t nmols,int32_t maxlingos,int32_t refbase) 
{ //{{{
    // Compares all molecules 0:nmols against molecules refbase:refbase+gridDim.x
    // Calculate nBlocks*nThreads tanimotos at a time and histograms results
    // with 0.01 resolution into 101 output bins [0, 0.01 ..., 0.99, 1.0] on a row-wise basis
    // Essentially, is the fusion of gpuLingoSim3_multirow and histogramMultipleRows101
    // Assumes histogram is already initialized, so we can accumulate results
    // sm11 version - uses atomic adds into global memory

    const int histogramBins = 101;

    // Places reference multiset into shared memory
    // Requires 4*len(ref multiset) 
    extern __shared__ int32_t shmem[];
    const int32_t refidx = refbase + blockIdx.x;
    if (threadIdx.x == 0) {
        shmem[0] = devRLingoLengths[refidx];
        shmem[1] = devRSetMagnitudes[refidx];
    }
    __syncthreads();
    const int32_t refLingoLength = shmem[0];
    const int32_t refSetMagnitude = shmem[1];
    __syncthreads();

    // Load the reference multiset
    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }

    // Each block processes exactly one row. Histogram immediately upon calculating a
    // new block of values.
    int32_t myOffset;
    float myTanimoto;
    unsigned index;
    int32_t* glHistogram = histogramRows + blockIdx.x*pitchInInt32;
    // Initialize histogram
    if (threadIdx.x < histogramBins) glHistogram[threadIdx.x] = 0;
    __threadfence_block();
    __syncthreads();

    for (int32_t base = 0; base < nmols; base += blockDim.x) {
        myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            myTanimoto = deviceMultisetTanimoto3_bT_tex2_mags(refLingos,refLingoCounts,refLingoLength,refSetMagnitude,devQSetMagnitudes[myOffset],devQLingoLengths[myOffset],maxlingos,myOffset);
            index = (unsigned)(myTanimoto*100);
            atomicAdd(glHistogram+index,1);
        }
    }
    return;
}
//}}}
"""

sm12_histogramming_kernels = """
__global__ void histogramMultipleRows101(float* tanimotoRows,int32_t pitchInFloat,int32_t nmols,int32_t* histogramRows,int32_t pitchInInt32) 
{ //{{{
    // 101 histogram bins (0, 0.01, ... 1.00) and six warps
    // sm12 version uses atomic adds to shared memory
    const int histogramBins = 101;
    __shared__ int32_t shHistogram[histogramBins];
    
    // Initialize local histogram
    if (threadIdx.x < histogramBins) shHistogram[threadIdx.x] = 0;
    __syncthreads();

    // Each block processes exactly one row
    const float* tanimotos = tanimotoRows + blockIdx.x*pitchInFloat;
    int32_t* glHistogram = histogramRows + blockIdx.x*pitchInInt32;

    // Iterate over elements in the row
    for (int base = 0; base < nmols; base += blockDim.x) {
        int myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            float data = tanimotos[myOffset];
            unsigned index = min((unsigned int)(data*100),101);
            atomicAdd(shHistogram+index,1);
        }
    }
    __syncthreads();

    // Write out to gmem
    if (threadIdx.x < histogramBins) glHistogram[threadIdx.x] = shHistogram[threadIdx.x];
    return;

} //}}}


__global__ void gpuHistogrammedLingo_multirow(int32_t* devRLingoLengths,int32_t* devQLingoLengths,
                                              int32_t* devRSetMagnitudes,int32_t* devQSetMagnitudes,
                                              int32_t* histogramRows,int32_t pitchInInt32,
                                              int32_t nmols,int32_t maxlingos,int32_t refbase) 
{ //{{{
    // Compares all molecules 0:nmols against molecules refbase:refbase+gridDim.x
    // Calculate nBlocks*nThreads tanimotos at a time and histograms results
    // with 0.01 resolution into 101 output bins [0, 0.01 ..., 0.99, 1.0] on a row-wise basis
    // Essentially, is the fusion of gpuLingoSim3_multirow and histogramMultipleRows101
    // Assumes histogram is already initialized, so we can accumulate results
    // sm12 version - uses atomic adds into shared memory

    const int histogramBins = 101;
    __shared__ int32_t shHistogram[histogramBins];
    if (threadIdx.x < histogramBins) shHistogram[threadIdx.x] = 0;

    // Places reference multiset into shared memory
    // Requires 4*len(ref multiset)
    extern __shared__ int32_t shmem[];
    const int32_t refidx = refbase + blockIdx.x;
    if (threadIdx.x == 0) {
        shmem[0] = devRLingoLengths[refidx];
        shmem[1] = devRSetMagnitudes[refidx];
    }
    __syncthreads();
    const int32_t refLingoLength = shmem[0];
    const int32_t refSetMagnitude = shmem[1];
    __syncthreads();

    // Load the reference multiset
    int32_t* refLingos = shmem;
    int32_t* refLingoCounts = refLingos + refLingoLength;
    for (int32_t base = 0; base < refLingoLength; base += blockDim.x) {
        if ((base + threadIdx.x) < refLingoLength) {
            refLingos[base+threadIdx.x] = tex2D(tex2RefLingoMatrix,base+threadIdx.x,refidx);
            refLingoCounts[base+threadIdx.x] = tex2D(tex2RefLingoCountMatrix,base+threadIdx.x,refidx);
        }
    }
    __syncthreads();

    // Each block processes exactly one row. Histogram immediately upon calculating a
    // new block of values.
    int32_t myOffset;
    float myTanimoto;
    unsigned index;
    int32_t* glHistogram = histogramRows + blockIdx.x*pitchInInt32;

    for (int32_t base = 0; base < nmols; base += blockDim.x) {
        myOffset = base + threadIdx.x;
        if (myOffset < nmols) {
            myTanimoto = deviceMultisetTanimoto3_bT_tex2_mags(refLingos,refLingoCounts,refLingoLength,refSetMagnitude,devQSetMagnitudes[myOffset],devQLingoLengths[myOffset],maxlingos,myOffset);
            index = (unsigned)(myTanimoto*100);
            atomicAdd(shHistogram+index,1);
        }
    }

    __syncthreads();
    // Write results out to gmem
    if (threadIdx.x < histogramBins) glHistogram[threadIdx.x] = shHistogram[threadIdx.x];
    return;
}
//}}}
"""
