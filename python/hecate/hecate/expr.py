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

hecate_dir = os.environ["HECATE"]
hecateBuild = hecate_dir+"/build"
heaan_keyset = "/heaan_keyset"
libpath = hecateBuild + "/lib/"
libname = libpath
osname = system()
if osname == 'Linux':
    libname = libpath + "libHecateFrontend.so"
elif osname == 'Darwin':
    libname = libpath + "libHecateFrontend.dylib"
else:
    raise UnsupportedPlatform
lt = ctypes.CDLL(libname)
os.environ['PATH'] = libpath + os.pathsep + os.environ['PATH']





"""Object Creation"""
lt.createConstant.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.createFunc.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, 
        ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.initFunc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
        ]
lt.createConstant.restype = ctypes.c_size_t
lt.createConstant.restype = ctypes.c_size_t
lt.createFunc.restype = ctypes.c_size_t

"""Immediate arguments"""
lt.createRotation.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t
        ]
lt.createRotation.restype = ctypes.c_size_t


"""Unary Operation"""
lt.createUnary.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t
        ]
lt.createUnary.restype = ctypes.c_size_t
"""Binary Operation"""
lt.createBinary.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t
        ]
lt.createBinary.restype = ctypes.c_size_t
"""Return"""
lt.setOutput.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), 
        ctypes.c_size_t]
"""compile"""
lt.save.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lt.save.restype = ctypes.c_char_p

lt.init.restype = ctypes.c_void_p
lt.finalize.argtypes = [ctypes.c_void_p]
"""Context Generation"""
ctxt = lt.init()
import sys
# weakref.finalize(sys.modules[__name__], lt.finalize, ctxt)

"""OpCode Tables"""
toUnary = {
        "bootstrap": 0,  # Bootstrap
        }
toBinary = {
        "add": 6,  # Addition
        "sub": 7,  # Subtraction
        "mul": 8,  # Multiplication
        }
toInnerUnary = {
        "neg": 13,  # Negation
        }



def save(dirs="", cst_dirs=""):
    (frame, filename, line_number, function_name, lines,
            index) = getProperFrame()

    start = time.perf_counter()
    proc_start = time.process_time()
    if dirs=="" :
        dirs = os.getcwd()
    if cst_dirs=="" :
        cst_dirs = os.getcwd()
    
    [func.eval() for func in funcList]
    name = filename.split("/")[-1].split(".")[0].encode('utf-8')
    name = (dirs +"/"+ filename.split("/")[-1].split(".")[0]+ ".mlir" ).encode('utf-8') 
    cst_name = (cst_dirs).encode('utf-8') 
    name = lt.save(ctxt, cst_name, name ).decode('utf-8')


    return name 

"""Set UnaryOperators"""


def unaryFactory(name, opcode):
    def unaryMethod(self):
        (frame, filename, line_number, function_name, lines,
                index) = inspect.stack()[1]
        return Expr(
                lt.createUnary(ctxt, opcode, self.obj, filename.encode('utf-8'),
                    line_number))

    globals()[name] = unaryMethod
[unaryFactory(name, opcode) for name, opcode in toUnary.items()]
"""Metaclasses"""


class hecateMetaBase(type):
    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

        def raiser():
            raise Exception("Copying Hecate object is forbidden")
        setattr(newcls, "__copy__", raiser)
        setattr(newcls, "__deepcopy__", raiser)
        return newcls


class hecateMetaBinary(hecateMetaBase):
    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

        def binaryFactory(cls, name, opcode):
            def binaryMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                return Expr(
                        lt.createBinary(ctxt, opcode, self.obj, tmp.obj,
                            filename.encode('utf-8'), line_number))

            def binaryReverseMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                return Expr(
                        lt.createBinary(ctxt, opcode, tmp.obj, self.obj,
                            filename.encode('utf-8'), line_number))

            def binaryInplaceMethod(self, other):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                tmp = resolveType(other)
                self.obj = lt.createBinary(ctxt, opcode, self.obj, tmp.obj,
                        filename.encode('utf-8'),
                        line_number)

            setattr(cls, f"__{name}__", binaryMethod)
            setattr(cls, f"__r{name}__", binaryReverseMethod)
            setattr(cls, f"__i{name}__", binaryReverseMethod)


        def innerFactory(cls, name, opcode):
            def innerMethod(self):
                (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
                return Expr(
                        lt.createUnary(ctxt, opcode, self.obj,
                            filename.encode('utf-8'), line_number))

            setattr(cls, f"__{name}__", innerMethod)

        [ 
            binaryFactory(newcls, name, opcode)
            for name, opcode in toBinary.items()
        ]
        [
            innerFactory(newcls, name, opcode)
            for name, opcode in toInnerUnary.items()
        ]

        def rotate(self, offset) :
            (frame, filename, line_number, function_name, lines,
                        index) = inspect.stack()[1]
            return Expr(lt.createRotation(ctxt, self.obj, offset, 
                filename.encode('utf-8'), line_number))
        setattr(newcls, "rotate", rotate)
        return newcls

"""Helper Functions"""

def getProperFrame():
    (frame, thisname, line_number, function_name, lines,
            index) = inspect.stack()[0]
    for (frame, filename, line_number, function_name, lines,
            index) in inspect.stack():
        if thisname != filename:
            return (frame, filename, line_number, function_name, lines, index)

def recType(li):
    if isinstance(li, int) or isinstance(li, float):
        return []
    elif isinstance(li, list):
        tys = [recType(x) for x in li]
        if all(i == tys[0] for i in tys):
            return [len(li)] + tys[0]
        else:
            raise Exception("Cannot create compatiable type")
    else:
        raise Exception("Cannot create compatiable type")


def flatten(L):
    for l in L:
        if isinstance(l, list):
            yield from flatten(l)
        else:
            yield l


def resolveType(other):
    if isinstance(other, Expr):
        return other
    elif isinstance(other, int):
        return Plain(np.array([other], dtype=np.float64)) #Plain([other])
    elif isinstance(other, float):
        return Plain(np.array([other], dtype=np.float64)) #Plain([other])
    elif isinstance(other, list):
        return Plain(np.array(other, dtype=np.float64)) #Plain(other)
    # elif isinstance(other, torch.Tensor) :
    #     return Plain( torch.flatten(other).tolist() )
    elif isinstance(other, np.ndarray):
        return Plain(other)
    else:
        raise Exception("Cannot create compatiable type")
    """Binary-operatable expression"""


class Expr(metaclass=hecateMetaBinary):
    def __init__(self, obj):
        self.obj = obj

class Plain(Expr):
    def __init__(self, data, scale = 40):
        # carr = (ctypes.c_double *
        #         len(data))(*[float(x) for x in flatten(data)])
        if not isinstance(data, np.ndarray) :
            data = np.array(data, dtype=np.float64)
         
        carr = npcl.as_ctypes(data) 
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        super().__init__(
                lt.createConstant(ctxt, carr, len(data), filename.encode('utf-8'),
                    line_number))

class Empty :
    def __init__ (self) : 
        pass
    def __add__(self, other) :
        return other
    def __radd__(self, other) :
        return other
    def __iadd__(self, other) :
        return other
    def __sub__(self, other) :
        return other
    def __rsub__(self, other) :
        return other
    def __isub__(self, other) :
        return other


"""Function Decorator"""
funcList = []


def func(param):
    def generateFunc(func):
        global funcList
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        a = Func(func, param, filename, line_number)
        funcList.append(a)
        return a

    return generateFunc


"""Function object"""
from collections.abc import Iterable

class Func(metaclass=hecateMetaBase):
    def __init__(self, fun, paramstr, filename, line_number):
        name = fun.__name__
        self.fun = fun

        arg = paramstr.split(",")
        inputs = [ a== "c" for a in arg]

        self.inputlen = len(inputs)
        inputTys = (ctypes.c_int * len(inputs))(*inputs)
        self.obj = lt.createFunc(ctxt, name.encode('utf-8'), inputTys,
                len(inputs), filename.encode('utf-8'),
                line_number)

    def eval(self):
        inputarr = (ctypes.c_size_t * self.inputlen)()
        lt.initFunc(ctxt, self.obj, inputarr, self.inputlen)
        inputs = [Expr(x) for x in inputarr[:self.inputlen]]
        returns = self.fun(*inputs)
        if not isinstance (returns, Iterable) : 
            returns = [returns]
        outputs = [ x.obj for  x in returns]
        outputvec = (ctypes.c_size_t * len(outputs))(*outputs)
        lt.setOutput(ctxt, self.obj, outputvec, len(outputs))

    def __call__(self, *args):
        (frame, filename, line_number, function_name, lines,
                index) = getProperFrame()
        tmps = [resolveType(arg) for arg in args]
        argarr = (ctypes.c_size_t * len(args))(*[tmp.obj for tmp in tmps])

        return Expr(
                lt.createCall(ctxt, self.obj, argarr, len(args),
                    filename.encode('utf-8'), line_number))


def removeCtxt() :
    lt.removeCtxt()
