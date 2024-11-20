

import ctypes
import weakref
import re
import inspect
from subprocess import Popen
from collections.abc import Iterable
from platform import system
# import torch

import os
import time
import numpy as np
import numpy.ctypeslib as npcl
from pathlib import Path



hecate_dir = Path(os.environ["HECATE"])
hecateBuild = hecate_dir / "build" 


if not hecateBuild.is_dir() : # We expect that this is library path 
    hecateBuild  = hecate_dir

libpath = hecateBuild / "lib"
libname = libpath
osname = system()
if osname == 'Linux':
    libname = libpath / "libSEAL_HEVM.so"
elif osname == 'Darwin':
    libname = libpath / "libSEAL_HEVM.dylib"
else:
    raise UnsupportedPlatform
lw = ctypes.CDLL(libname)
os.environ['PATH'] = str(libpath) + os.pathsep + os.environ['PATH']


# Init VM functions
lw.initFullVM.argtypes = [ctypes.c_char_p]
lw.initFullVM.restype = ctypes.c_void_p 
lw.initClientVM.argtypes = [ctypes.c_char_p]
lw.initClientVM.restype = ctypes.c_void_p 
lw.initServerVM.argtypes = [ctypes.c_char_p]
lw.initServerVM.restype = ctypes.c_void_p 

# Init SEAL Contexts
lw.create_context.argtypes = [ctypes.c_char_p]
lw.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lw.loadClient.argtypes = [ctypes.c_void_p,  ctypes.c_void_p]
lw.getArgLen.argtypes = [ctypes.c_void_p]
lw.getArgLen.restype = ctypes.c_int64
lw.getResLen.argtypes = [ctypes.c_void_p]
lw.getResLen.restype = ctypes.c_int64

# Encrypt/Decrypt Functions
lw.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lw.decrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
lw.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

# Helper Functions for ciphertext access
lw.getResIdx.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lw.getResIdx.restype = ctypes.c_int64 
lw.getCtxt.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lw.getCtxt.restype = ctypes.c_void_p 

# Runner Functions
lw.preprocess.argtypes = [ctypes.c_void_p]
lw.run.argtypes = [ctypes.c_void_p]

#Debug Function
lw.setDebug.argtypes = [ctypes.c_void_p, ctypes.c_bool]


class HEVM : 
    def __init__ (self, path = str((Path.home() / ".hevm" / "seal").absolute()) , option= "full") : 
        self.option = option
        if not Path(path).is_dir() : 
            print ("Press Any key to generate SEAL files (or just kill with ctrl+c)")
            input()
            Path(path).mkdir(parents=True)
            lw.create_context(path.encode('utf-8'))

        if option == "full" : 
            self.vm = lw.initFullVM(path.encode('utf-8'))
        elif option == "client" :
            self.vm = lw.initClientVM(path.encode('utf-8'))
        elif  option == "server" :
            self.vm = lw.initServerVM(path.encode('utf-8'))

    # def load (self, func,   preprocess=True, const_path =str( (Path(func_dir) / "_hecate_{func}.cst").absoluate() ), hevm_path = str(Path(func_dir) / "_hecate_{func}.hevm"), func_dir = str(Path.cwd()), ) :
    def load (self, const_path, hevm_path, preprocess=True) :
        if not Path(const_path).is_file() :
            raise Exception(f"No file exists in const_path {const_path}")
        if not Path(hevm_path).is_file() :
            raise Exception(f"No file exists in hevm_path {hevm_path}")

        if self.option == "full" or self.option == "server" :
            lw.load(self.vm, const_path.encode('utf-8'), hevm_path.encode('utf-8'))
        elif self.option ==  "client" :
            lw.loadClient (self.vm, const_path.encode('utf-8'))
        if (preprocess) :
            lw.preprocess (self.vm)
        else :
            raise Exception("Not implemented in SEAL_HEVM")

        self.arglen = lw.getArgLen(self.vm)
        self.reslen = lw.getResLen(self.vm)

    def run (self) : 
        lw.run(self.vm)

    def setInput(self, i, data) :
        if not isinstance(data, np.ndarray) :
            data = np.array(data, dtype=np.float64)
        carr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        print(data)
        lw.encrypt(self.vm, i, carr, len(data))

    def setDebug (self, enable) : 
        lw.setDebug(self.vm, enable)


    def getOutput (self) : 
        result = np.zeros( (self.reslen, 1 << 14), dtype=np.float64)
        data = np.zeros(  1 << 14, dtype=np.float64)
        for i in range(self.reslen) :
            # carr = npcl.as_ctypes(data) 
            carr =  data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lw.decrypt_result(self.vm, i, carr)
            # result[i] = npcl.as_array(carr, shape= 1<<14)
            result[i] = data

        return result







