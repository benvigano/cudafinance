import pytest
import numpy as np
from src.cudafinance.cuda_module import launchSMA
from src.cudafinance.cuda_module import launchMomentum

def test_sma_basic():
    input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3
    expected_output = np.array([1, 1.5, 2, 3, 4], dtype=np.float32)
    
    output = np.zeros_like(input_data)
    launchSMA(input_data, output, window_size)
    
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_sma_constant():
    input_data = np.array([2, 2, 2, 2, 2], dtype=np.float32)
    window_size = 3
    expected_output = np.array([2, 2, 2, 2, 2], dtype=np.float32)
    
    output = np.zeros_like(input_data)
    launchSMA(input_data, output, window_size)
    
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_sma_larger_window():
    input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 10
    expected_output = np.array([1, 1.5, 2, 2.5, 3], dtype=np.float32)
    
    output = np.zeros_like(input_data)
    launchSMA(input_data, output, window_size)
    
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_sma_empty():
    input_data = np.array([], dtype=np.float32)
    window_size = 3
    expected_output = np.array([], dtype=np.float32)
    
    output = np.array([], dtype=np.float32)
    launchSMA(input_data, output, window_size)
    
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_sma_large_array():
    np.random.seed(0)
    input_data = np.random.randn(1000).astype(np.float32)
    window_size = 100

    output = np.zeros_like(input_data)
    launchSMA(input_data, output, window_size)

    expected_output = np.convolve(input_data, np.ones(window_size), 'valid') / window_size
    expected_output = np.pad(expected_output, (window_size-1, 0), mode='edge')

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_basic():
    input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    window_size = 3

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    for i in range(window_size, len(input_data)):
        expected_output[i] = input_data[i] - input_data[i - window_size]

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_large_array():
    np.random.seed(0)
    input_data = np.random.randn(1000).astype(np.float32)
    window_size = 50

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    for i in range(window_size, len(input_data)):
        expected_output[i] = input_data[i] - input_data[i - window_size]

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_small_window():
    input_data = np.array([5, 10, 15, 20, 25], dtype=np.float32)
    window_size = 1

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    for i in range(window_size, len(input_data)):
        expected_output[i] = input_data[i] - input_data[i - window_size]

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_large_window():
    input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    window_size = 15  # Larger than the input size

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_boundary_conditions():
    input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_size = 3

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    for i in range(window_size, len(input_data)):
        expected_output[i] = input_data[i] - input_data[i - window_size]

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

def test_momentum_negative_values():
    input_data = np.array([-5, -10, -15, -20, -25, -30, -35], dtype=np.float32)
    window_size = 2

    output = np.zeros_like(input_data)
    launchMomentum(input_data, output, window_size)

    expected_output = np.zeros_like(input_data)
    for i in range(window_size, len(input_data)):
        expected_output[i] = input_data[i] - input_data[i - window_size]

    np.testing.assert_allclose(output, expected_output, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
