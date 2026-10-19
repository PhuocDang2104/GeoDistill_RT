import ast
import ctypes as C
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import benchmark as B
from cuda_runtime import CUDA


class Contracts(unittest.TestCase):
    def test_nonfinite_and_fixed_parity(self):
        x = np.zeros(5, np.float32)
        self.assertTrue(B.parity(x, x)['passed'])
        self.assertFalse(B.parity(x+np.nan, x)['passed'])
        self.assertFalse(B.parity(x+.002, x)['passed'])
        self.assertFalse(B.parity(np.array([.061]+[0.]*100), np.zeros(101))['passed'])

    def test_stats(self):
        report = B.stats([4., 4., 4.])
        self.assertEqual(report['median_ms'], 4.)
        self.assertEqual(report['fps_from_median'], 250.)

    def test_shapes_and_units(self):
        arrays = {k: np.zeros(shape, np.float32) for k, shape in {**B.INPUTS, 'reference': B.OUTPUT}.items()}
        B.validate_sample(arrays)
        arrays['rgb'][0, 0, 0, 0] = 255
        with self.assertRaises(ValueError): B.validate_sample(arrays)

    def test_pointer_arguments_and_error_reporting(self):
        class Fn:
            def __init__(self, name): self.name = name
            def __call__(self, *args):
                if self.name == 'cudaGetErrorString': return b'mocked CUDA failure'
                if self.name == 'cudaMalloc': C.cast(args[0], C.POINTER(C.c_void_p))[0] = C.c_void_p(123456789012)
                return 0
        class Lib:
            def __getattr__(self, name):
                value = Fn(name);setattr(self, name, value);return value
        with patch('cuda_runtime.C.CDLL', return_value=Lib()):
            cuda = CUDA()
            self.assertEqual(cuda.malloc(16).value, 123456789012)
            self.assertEqual(cuda.functions['cudaMemcpyAsync'].argtypes[0], C.c_void_p)
            self.assertEqual(cuda.functions['cudaGraphInstantiateWithFlags'].argtypes[-1], C.c_ulonglong)
            cuda.functions['cudaStreamSynchronize'] = lambda *args: 1
            with self.assertRaisesRegex(RuntimeError, 'mocked CUDA failure'):
                cuda.call('cudaStreamSynchronize', C.c_void_p(1))
            cuda.close()

    def test_python_syntax(self):
        for path in Path(__file__).parent.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
