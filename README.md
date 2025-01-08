# cudafinance
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen?logo=nvidia&style=for-the-badge)](https://developer.nvidia.com/cuda-zone)

Cudafinance is a Python package built on top of CUDA. It leverages GPU acceleration for computation of financial indicators.

## Installation
> [!NOTE]
> Cudafinance is currently only available for manual installation and is not yet published on PyPI due to challenges in the process of building via Git workflow. (find details in Development section below)
### Manual installation
#### Prerequisites
- Python 3.6+
- CUDA Toolkit
- Build tools: GCC, G++, CMake (`sudo apt install build-essential cmake python3-dev`)
- Python libraries: `pybind11`, `numpy`, `pytest` (`python3 -m pip install pybind11 numpy pytest`).
#### Installation
- Clone the repo: `git clone https://github.com/benvigano/cudafinance/ cudafinance`, `cd cudafinance`
- Set CUDA path:
  ```bash
  export CUDA_HOME=/path/to/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- Build: `python3 setup.py build_ext --inplace`
- Test: `pytest tests/`
## Benchmarking
<p align="start">
    <img src="https://github.com/user-attachments/assets/44671072-0efc-400d-a5a1-1c2507ce35c2" alt="Home page screen capture" style="width: 500px; height: auto;">
</p>
- 20x speed compared to Pandas implementation (already optimized with Cython and cumulative sums).

## Support
### FIR filters
Cudafinance currently **only includes indicators that are FIR filters (Finite Impulse Response)**. This is due to the parallelizable nature of FIR filters, given their dependency on a finite number of input samples.
### IIR filters
IIR filters (Infinite Impulse Response) are currently **not supported** due to their inherent recursive structure which causes them not to be directly optimizable via parallelization. (see Development section below)

## Development
- **IIRs**: In future updates I plan to **add support** for IIR-based indicators employing hybrid approaches (splitting computations into parallel batches or approximating IIR behavior with FIR filters).
- **PyPI**: If building via GitHub Workflow proves not feasible, I will consider pre-building for a few common platforms and uploading the wheels directly to PyPI.
