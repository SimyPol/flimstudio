from __future__ import annotations

import ctypes as ct
from pathlib import Path
import sys

import numpy as np

MATLAB_BIN = Path(r"C:\Program Files\MATLAB\R2025b\bin\win64")
if hasattr(os := __import__("os"), "add_dll_directory"):
    os.add_dll_directory(str(MATLAB_BIN))
mat = ct.CDLL(str(MATLAB_BIN / "libmat.dll"))
mx = ct.CDLL(str(MATLAB_BIN / "libmx.dll"))

mat.matOpen.argtypes = [ct.c_char_p, ct.c_char_p]
mat.matOpen.restype = ct.c_void_p
mat.matGetDir.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
mat.matGetDir.restype = ct.POINTER(ct.c_char_p)
mat.matGetVariable.argtypes = [ct.c_void_p, ct.c_char_p]
mat.matGetVariable.restype = ct.c_void_p
mat.matClose.argtypes = [ct.c_void_p]
mat.matClose.restype = ct.c_int
mx.mxGetNumberOfDimensions_800.argtypes = [ct.c_void_p]
mx.mxGetNumberOfDimensions_800.restype = ct.c_size_t
mx.mxGetDimensions_800.argtypes = [ct.c_void_p]
mx.mxGetDimensions_800.restype = ct.POINTER(ct.c_size_t)
mx.mxGetNumberOfElements_800.argtypes = [ct.c_void_p]
mx.mxGetNumberOfElements_800.restype = ct.c_size_t
mx.mxGetData_800.argtypes = [ct.c_void_p]
mx.mxGetData_800.restype = ct.c_void_p
mx.mxGetClassID_800.argtypes = [ct.c_void_p]
mx.mxGetClassID_800.restype = ct.c_int
mx.mxDestroyArray.argtypes = [ct.c_void_p]

CLASS = {6: np.float64, 7: np.float32, 8: np.int8, 9: np.uint8, 10: np.int16, 11: np.uint16, 12: np.int32, 13: np.uint32, 14: np.int64, 15: np.uint64}


def read(path: Path):
    handle = mat.matOpen(str(path).encode(), b"r")
    if not handle:
        raise OSError(path)
    count = ct.c_int()
    names_ptr = mat.matGetDir(handle, ct.byref(count))
    names = [names_ptr[i].decode() for i in range(count.value)]
    arrays = {}
    for name in names:
        ptr = mat.matGetVariable(handle, name.encode())
        class_id = mx.mxGetClassID_800(ptr)
        if class_id in CLASS:
            nd = mx.mxGetNumberOfDimensions_800(ptr)
            dims_ptr = mx.mxGetDimensions_800(ptr)
            shape = tuple(int(dims_ptr[i]) for i in range(nd))
            n = int(mx.mxGetNumberOfElements_800(ptr))
            dtype = np.dtype(CLASS[class_id])
            address = mx.mxGetData_800(ptr)
            buf = (ct.c_byte * (n * dtype.itemsize)).from_address(address)
            arrays[name] = np.frombuffer(buf, dtype=dtype, count=n).reshape(shape, order="F").copy()
        mx.mxDestroyArray(ptr)
    mat.matClose(handle)
    return arrays


if __name__ == "__main__":
    source = Path(sys.argv[1])
    data = read(source)
    print(source.name)
    for key, value in data.items():
        print(key, value.shape, value.dtype, float(np.min(value)), float(np.max(value)), float(np.sum(value)))
    if len(sys.argv) > 2:
        np.savez_compressed(sys.argv[2], **data)
