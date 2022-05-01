import torch
from dataclasses import dataclass

@dataclass
class NystromPreconditioner:
    x: torch.Tensor
