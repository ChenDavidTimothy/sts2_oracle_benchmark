from __future__ import annotations
import torch
from raster import has_cupy_cuda


def main():
    print("torch:", torch.__version__)
    print("torch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("compute capability:", torch.cuda.get_device_capability(0))
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA tensor test:", float(y[0, 0]))
    print("CuPy raster backend:", has_cupy_cuda())

if __name__ == "__main__":
    main()
