import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from src.cudafinance.cuda_module import launchSMA


def benchmark_sma_range():
    sizes = [100_000, 250_000, 500_000, 1_000_000, 5_000_000] + list(range(10_000_000, 100_000_000, 10_000_000))
    window_size = 100
    pandas_times = []
    cuda_times = []

    for num_elements in sizes:
        print(f"Benchmarking for n={num_elements}...")

        # Generate random input data
        np.random.seed(42)
        input_data = np.random.randn(num_elements).astype(np.float32)

        # Pandas SMA
        start_time = time.time()
        series = pd.Series(input_data)
        sma_pandas = series.rolling(window=window_size, min_periods=1).mean().values
        pandas_time = time.time() - start_time
        pandas_times.append(pandas_time)

        # cudafinance SMA
        output_cuda = np.zeros_like(input_data)
        start_time = time.time()
        launchSMA(input_data, output_cuda, window_size)
        cuda_time = time.time() - start_time
        cuda_times.append(cuda_time)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, pandas_times, label="Pandas SMA", marker="o", linestyle="-")
    plt.plot(sizes, cuda_times, label="CUDA SMA", marker="o", linestyle="--")
    plt.xlabel("Number of Elements (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Pandas vs CUDA SMA Performance")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save plot as .pdf
    plt.savefig("sma_performance_comparison.pdf")
    print("Benchmark results saved to 'sma_performance_comparison.pdf'.")


if __name__ == "__main__":
    benchmark_sma_range()
