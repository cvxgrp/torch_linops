from linops import LinearOperator
import linops as lo
import torch

class FusedOperator(LinearOperator):
    def __init__(self, shape: tuple[int, int],
                       in_slices: list[slice],
                       out_slices: list[slice],
                       ops: list[LinearOperator],
                       adjoint=None):
        self._shape = shape
        self._in_slices = in_slices
        self._out_slices = out_slices
        self._ops = ops
        if adjoint is None:
            self._adjoint = FusedOperator((shape[1], shape[0]), out_slices,
                    in_slices, ops, self)
        else:
            self._adjoint = adjoint
    
    def _matmul_impl(self, v):
        out = torch.zeros(self._shape[0], device=v.device, dtype=v.dtype)
        for in_s, out_s, op in zip(self._in_slices, self._out_slices, self._ops,
                # strict=True, Readd this when upgrading MPV to 3.10 
                ):
            out[out_s] += op @ v[in_s]
        return out

class ZeroOperator(LinearOperator):
    def __init__(self, shape: tuple[int, int], adjoint=None):
        self._shape = shape
        if adjoint is None:
            self._adjoint = ZeroOperator((shape[1], shape[0]), self)
        else:
            self._adjoint = adjoint
    
    def _matmul_impl(self, v):
        return torch.zeros(self._shape[0], device=v.device, dtype=v.dtype)


def vstack(tup):
    tup = [lo.aslinearoperator(A) for A in tup]
    left_shape = [A.shape[0] for A in tup] 
    right_shape = {A.shape[1] for A in tup}
    assert len(right_shape) == 1
    right_dim = next(iter(right_shape))
    left_dim = sum(left_shape)
    value = 0

    return FusedOperator(
        (left_dim, right_dim),
        [slice(None) for _ in tup],
        [slice(value, (value := value + size)) for size in left_shape],
        tup)

def hstack(tup):
    tup = [lo.aslinearoperator(A) for A in tup]
    left_shape = {A.shape[0] for A in tup} 
    right_shape = [A.shape[1] for A in tup]
    assert len(left_shape) == 1
    left_dim = next(iter(left_shape))
    right_dim = sum(right_shape)
    value = 0

    return FusedOperator(
        (left_dim, right_dim),
        [slice(value, (value := value + size)) for size in right_shape],
        [slice(None) for _ in tup],
        tup)

def bmat(blocks):
    """
    Optimizes out `ZeroOperator`s.
    """
    blocks = [[lo.aslinearoperator(A) for A in r] for r in blocks]

    raise NotImplementedError("bmat implementation is incomplete")
