# cudafinance
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen?logo=nvidia&style=for-the-badge)](https://developer.nvidia.com/cuda-zone)

Cudafinance is a Python package built on top of CUDA. It leverages GPU acceleration for computation of financial indicators.

## Installation
> [!NOTE]
> The package is not yet available on PyPI due to complications in the process of building via Git workflow. If I can't find a way out I will consider pre-building for a few common platforms and uploading the wheels directly to PyPI.

## Support
### FIR filters
Cudafinance currently **only includes indicators that are FIR filters (Finite Impulse Response)**. This is due to the parallelizable nature of FIR filters, given their dependency on a finite number of input samples.
### IIR filters
IIR filters (Infinite Impulse Response) are not currently supported due to their inherent recursive structure which causes them not to be directly optimizable via parallelization. (see Development section below)

## Development
In future updates I plan to **add support** for IIR-based indicators employing hybrid approaches (splitting computations into parallel batches or approximating IIR behavior with FIR filters).
