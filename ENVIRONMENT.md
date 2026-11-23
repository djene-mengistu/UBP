# Reference environment

The completed reference runs were executed on Linux with an NVIDIA GeForce RTX 4090, Python 3.11.5, PyTorch 2.1.0+cu118, NumPy 1.24.3, OpenCV 4.8.1, h5py 3.9.0, SciPy 1.11.1, and Matplotlib 3.7.2. The model dependencies included timm 1.0.14, einops 0.8.0, ml-collections 1.1.0, and PyYAML 6.0.

Install the PyTorch build that matches your CUDA driver separately; `requirements.txt` lists the other direct runtime/plotting dependencies. These are observed reference versions, not a guarantee that every later package release is compatible. The run profiles include device, precision, latency scope, parameter counts, and profiler coverage.
