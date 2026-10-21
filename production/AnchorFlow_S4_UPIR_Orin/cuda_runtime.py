"""Small CUDA12+ runtime binding. No PyTorch, PyCUDA or cuda-python dependency.

All host transfers occur outside capture/timing. Device staging is timed.
"""
import ctypes as C
import ctypes.util


class CUDA:
    def __init__(self):
        path = ctypes.util.find_library('cudart') or 'libcudart.so.12'
        self.lib = C.CDLL(path)
        P, S, I = C.c_void_p, C.c_size_t, C.c_int
        self.functions = {}
        signatures = {
            'cudaGetErrorString': (C.c_char_p, [I]),
            'cudaMalloc': (I, [C.POINTER(P), S]), 'cudaFree': (I, [P]),
            'cudaMemcpy': (I, [P, P, S, I]),
            'cudaMemcpyAsync': (I, [P, P, S, I, P]),
            'cudaStreamCreateWithFlags': (I, [C.POINTER(P), C.c_uint]),
            'cudaStreamSynchronize': (I, [P]), 'cudaStreamDestroy': (I, [P]),
            'cudaStreamBeginCapture': (I, [P, I]),
            'cudaStreamEndCapture': (I, [P, C.POINTER(P)]),
            'cudaGraphInstantiateWithFlags': (I, [C.POINTER(P), P, C.c_ulonglong]),
            'cudaGraphLaunch': (I, [P, P]), 'cudaGraphExecDestroy': (I, [P]),
            'cudaGraphDestroy': (I, [P]),
            'cudaEventCreate': (I, [C.POINTER(P)]),
            'cudaEventRecord': (I, [P, P]),
            'cudaEventElapsedTime': (I, [C.POINTER(C.c_float), P, P]),
            'cudaEventDestroy': (I, [P]),
            'cudaMemGetInfo': (I, [C.POINTER(S), C.POINTER(S)]),
            'cudaRuntimeGetVersion': (I, [C.POINTER(I)]),
        }
        for name, (restype, argtypes) in signatures.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            self.functions[name] = fn
        self.allocations, self.resources = [], []

    def call(self, name, *args):
        code = self.functions[name](*args)
        if code:
            detail = self.functions['cudaGetErrorString'](code).decode()
            raise RuntimeError(f'{name}: CUDA {code}: {detail}')

    def handle(self, create, destroy, *args):
        ptr = C.c_void_p()
        self.call(create, C.byref(ptr), *args)
        self.resources.append((destroy, ptr))
        return ptr

    def malloc(self, size):
        ptr = C.c_void_p()
        self.call('cudaMalloc', C.byref(ptr), int(size))
        self.allocations.append(ptr)
        return ptr

    def upload(self, array):
        ptr = self.malloc(array.nbytes)
        self.call('cudaMemcpy', ptr, array.ctypes.data, array.nbytes, 1)
        return ptr

    def memory(self):
        free, total = C.c_size_t(), C.c_size_t()
        self.call('cudaMemGetInfo', C.byref(free), C.byref(total))
        return {'free_mib': free.value/2**20, 'total_mib': total.value/2**20}

    def close(self):
        # Call only after stream synchronization, engine/context release.
        for destroy, ptr in reversed(self.resources):
            self.call(destroy, ptr)
        for ptr in reversed(self.allocations):
            self.call('cudaFree', ptr)
        self.resources.clear()
        self.allocations.clear()
