#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern void launchSMA_CUDA(const float* h_input, float* h_output, int numElements, int windowSize);

void launchSMA(py::array_t<float> input, py::array_t<float> output, int windowSize) {
    py::buffer_info input_buf = input.request(), output_buf = output.request();
    int numElements = input_buf.shape[0];

    // Ensure input and output are both properly sized
    if (output_buf.shape[0] != numElements) {
        throw std::runtime_error("Input and output arrays must have the same size.");
    }

    // Call the CUDA function
    launchSMA_CUDA(static_cast<float*>(input_buf.ptr), static_cast<float*>(output_buf.ptr), numElements, windowSize);
}

PYBIND11_MODULE(cuda_module, m) {
    m.def("launchSMA", &launchSMA, "Launches the Simple Moving Average (SMA) computation on CUDA.");
}
