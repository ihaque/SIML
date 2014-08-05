#!/usr/bin/python

import string
import time
import numpy
import sys
from pysiml.compiler import SMILEStoMatrices,cSMILEStoMatrices,preprocessNumbers
from pysiml.CPULingo import CPULingo

gpuEnabled = False
oeEnabled  = False
oclEnabled  = False

class timer:
    def __init__(self):
        self.st = None
        self.et = None
    def start(self):
        self.st = time.time()
        return
    def stop(self):
        self.et = time.time()
        return
    def elapsedms(self):
        return 1000*(self.et-self.st)


def main():
    f = open(sys.argv[1],"r")
    try:
        limit = int(sys.argv[2])
    except IndexError:
        limit = None
    smiles = []
    if limit is not None:
        for i in range(limit):
            smiles.append(f.readline())
    else:
        smiles = f.readlines()
    f.close()
    
    nmols = len(smiles)
    
    if gpuEnabled:
        gpusim = GPULingo()
    
    if oclEnabled:
        platform = pyopencl.get_platforms()[0]
        device0 = platform.get_devices(pyopencl.device_type.GPU)[0]
        print "Using OpenCL GPU",device0.get_info(pyopencl.device_info.NAME)
        print
        oclsim = OCLLingo(device0)
    
    cpusim = CPULingo()

    T = timer()
    T.start()
    (lpy,cpy,tpy,mpy,npy) = SMILEStoMatrices(smiles)
    T.stop()
    pycompileTime = T.elapsedms()
    T.start()
    (lc,cc,tc,mc,nc) = cSMILEStoMatrices(smiles)
    T.stop()
    ccompileTime = T.elapsedms()


    print "SMILES compiler benchmarks:"
    print "---------------------------"
    print "Took %.2f ms to convert %d SMILES to Numpy format (Python)"%(pycompileTime,nmols)
    print "Took %.2f ms to convert %d SMILES to Numpy format (C)"%(ccompileTime,nmols)
    print

    if oeEnabled:
        smioe = []
        for smi in smiles:
            f = smi.split()
            smioe.append(preprocessNumbers(f[0]))

        
    
    print "Initialization benchmarks:"
    print "--------------------------"

    if gpuEnabled:
        T.start()
        # We need to use the Python matrices if comparing to OpenEye because the C and the Python preprocess things differently
        if oeEnabled:
            gpusim.set_refsmiles(lpy,cpy,tpy,mpy)
            gpusim.set_qsmiles(lpy,cpy,tpy,mpy)
        else:
            gpusim.set_refsmiles(lc,cc,tc,mc)
            gpusim.set_qsmiles(lc,cc,tc,mc)
        T.stop()
        gpusetupTime = T.elapsedms()
        print "Took %.2f ms to initialize GPULingo object with SMILES matrices"%gpusetupTime

    if oclEnabled:
        T.start()
        # We need to use the Python matrices if comparing to OpenEye because the C and the Python preprocess things differently
        if oeEnabled:
            oclsim.set_refsmiles(lpy,cpy,tpy,mpy)
            oclsim.set_qsmiles(lpy,cpy,tpy,mpy)
        else:
            oclsim.set_refsmiles(lc,cc,tc,mc)
            oclsim.set_qsmiles(lc,cc,tc,mc)
        T.stop()
        oclsetupTime = T.elapsedms()
        print "Took %.2f ms to initialize OCLLingo object with SMILES matrices"%oclsetupTime

    T.start()
    if oeEnabled:
        cpusim.set_refsmiles(lpy,cpy,tpy,mpy)
        cpusim.set_qsmiles(lpy,cpy,tpy,mpy)
    else:
        cpusim.set_refsmiles(lc,cc,tc,mc)
        cpusim.set_qsmiles(lc,cc,tc,mc)
    T.stop()
    cpusetupTime = T.elapsedms()

    print "Took %.2f ms to initialize CPULingo object with SMILES matrices"%cpusetupTime
    print

    rowblocksize = 256

    print "Tanimoto calculation benchmarks:"
    print "--------------------------------"
    if gpuEnabled:
        gpuN2 = numpy.empty((nmols,nmols),dtype=numpy.float32)
        T.start()
        for base in range(0,nmols,rowblocksize):
            bound = min(base+rowblocksize,nmols)
            gpuN2[base:bound,:] = gpusim.getMultipleRows(base,bound)
        T.stop()
        gpuN2Time = T.elapsedms()
        print "Took %.2f ms to calculate %dx%d N^2 similarity matrix with CUDA GPU (blocksize = %d, synchronous)"%(gpuN2Time,nmols,nmols,rowblocksize)

    if oclEnabled:
        oclN2 = numpy.empty((nmols,nmols),dtype=numpy.float32)
        T.start()
        for base in range(0,nmols,rowblocksize):
            bound = min(base+rowblocksize,nmols)
            oclN2[base:bound,:] = oclsim.getMultipleRows(base,bound)
        T.stop()
        oclN2Time = T.elapsedms()
        print "Took %.2f ms to calculate %dx%d N^2 similarity matrix with OpenCL GPU (blocksize = %d, synchronous)"%(oclN2Time,nmols,nmols,rowblocksize)

    cpuN2 = numpy.empty((nmols,nmols),dtype=numpy.float32)
    T.start()
    for base in range(0,nmols,rowblocksize):
        bound = min(base+rowblocksize,nmols)
        cpuN2[base:bound,:] = cpusim.getMultipleRows(base,bound)
    T.stop()
    cpuN2Time = T.elapsedms()

    print "Took %.2f ms to calculate %dx%d N^2 similarity matrix with CPU (blocksize = %d, serial)"%(cpuN2Time,nmols,nmols,rowblocksize)

    if cpusim.supportsParallel():
        cpuPN2 = numpy.empty((nmols,nmols),dtype=numpy.float32)
        nprocs = 4 
        T.start()
        for base in range(0,nmols,rowblocksize):
            bound = min(base+rowblocksize,nmols)
            cpuPN2[base:bound,:] = cpusim.getMultipleRows(base,bound,nprocs=nprocs)
        T.stop()
        cpuPN2Time = T.elapsedms()
    
        print "Took %.2f ms to calculate %dx%d N^2 similarity matrix with CPU (blocksize = %d, parallel cores = %d)"%(cpuPN2Time,nmols,nmols,rowblocksize,nprocs)

    if oeEnabled:
        oeN2 = numpy.empty((nmols,nmols),dtype=numpy.float32)
        T.start()
        for base in range(0,nmols,rowblocksize):
            bound = min(base+rowblocksize,nmols)
            oeN2[base:bound,:] = oeGetMultipleRows(base,bound,smioe)
        T.stop()
        oeN2Time = T.elapsedms()

        print "Took %.2f ms to calculate %dx%d N^2 similarity matrix with OE-Python-naive (blocksize = %d)"%(oeN2Time,nmols,nmols,rowblocksize)

    print

    #for i in xrange(oclN2.shape[0]):
    #    for j in xrange(oclN2.shape[1]):
    #        if oclN2[i,j] != cpuN2[i,j] and abs(oclN2[i,j]-cpuN2[i,j]) >= 1e-6:
    #            print "OCL != CPU at (%d,%d): %f vs %f, abs diff = %g"%(i,j,oclN2[i,j],cpuN2[i,j],abs(oclN2[i,j]-cpuN2[i,j]))
    #print oclN2[0,:]
    #print cpuN2[0,:]

    print "Correctness checks (all should be True):"
    print "----------------------------------------"
    if gpuEnabled:
        print "GPU == CPU:",numpy.all(gpuN2 == cpuN2)
        #if cpusim.supportsParallel():
        #    print "GPU == CPU-parallel:",numpy.all(gpuN2 == cpuPN2)
    if oclEnabled:
        print "OCL == CPU:",numpy.all(numpy.abs(oclN2 - cpuN2) < 1e-6)
        #if cpusim.supportsParallel():
        #    print "OCL == CPU-parallel:",numpy.all(numpy.abs(oclN2 - cpuPN2) < 1e-6)
    if cpusim.supportsParallel():
        print "CPU == CPU-parallel:",numpy.all(cpuN2 == cpuPN2)
    if oeEnabled:
        if gpuEnabled:
            print "OE == GPU:",numpy.all(oeN2 == gpuN2)
        print "OE == CPU:",numpy.all(oeN2 == cpuN2)
        #if cpusim.supportsParallel():
        #    print "OE == CPU-parallel:",numpy.all(oeN2 == cpuPN2)

    return

if __name__ == "__main__":
    print "--------------------------------------------------"
    print "Benchmark and correctness check example for PySIML"
    print "--------------------------------------------------"
    print
    if len(sys.argv) < 2:
        print "Usage: %s [SMILES file] [optional limit to number of SMILES to read]"%sys.argv[0]
        sys.exit(1)
    try:
        from openeye.oechem import OELingoSim
        sim = OELingoSim()
        oeEnabled = True
        def oeGetMultipleRows(base,bound,smiles):
            block = numpy.empty((bound-base,len(smiles)),dtype=numpy.float32)
            oel = OELingoSim()
            for row in range(base,bound):
                oel.Init(smiles[row])
                for col in range(len(smiles)):
                    block[row-base,col] = oel.Similarity(smiles[col])
            return block
        print "Successfully loaded OpenEye OELingoSim, enabling comparison"
    except:
        print "Unable to load OpenEye libraries; no comparison to OpenEye\n"
        pass
    try:
        from pysiml.GPULingo import GPULingo
        gpuEnabled = True
        print "Successfully loaded GPULingo libraries; enabling CUDA GPU calculation"
    except ImportError:
        print "Unable to load GPULingo library; disabling CUDA GPU LINGO calculation"
    try:
        import pyopencl
        from pysiml.OCLLingo import OCLLingo
        oclEnabled = True
        print "Successfully loaded OCLLingo libraries; enabling OpenCL GPU calculation"
    except ImportError:
        print "Unable to load OCLLingo library; disabling OpenCL GPU LINGO calculation"
    csim = CPULingo()
    if csim.supportsParallel():
        print "pySIML built with OpenMP support; will test parallel CPULingo calculations"
    else:
        print "pySIML built without OpenMP support; will not test parallel CPULingo calculations"

    print
    main()
