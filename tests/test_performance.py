import time

import numpy as np
import pytest


@pytest.mark.performance
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [
        (5_000, 50),
        (10_000, 50),
        (20_000, 50),
    ],
)
def test_rust_adjacency_performance(n_rows, n_cols):
    """Benchmark Rust/Rayon adjacency calculation at multiple scales."""
    from likelihood import rust_py_integration

    rng = np.random.default_rng(42)

    data = rng.random((n_rows, n_cols), dtype=np.float64)

    similarity = n_cols - 5
    threshold = 0.05

    pair_count = n_rows * (n_rows - 1) // 2

    start = time.perf_counter()

    adjacency = rust_py_integration.cal_adjacency_matrix(
        data=data,
        n_cols=n_cols,
        similarity=similarity,
        threshold=threshold,
    )

    elapsed = time.perf_counter() - start

    connections = int(np.sum(adjacency) / 2)

    print("\n" + "=" * 60)
    print("Rust/Rayon Performance Benchmark")
    print("=" * 60)
    print(f"Input:             {n_rows:,} x {n_cols}")
    print(f"Output:            {adjacency.shape}")
    print(f"Pair comparisons:  {pair_count:,}")
    print(f"Elapsed time:      {elapsed:.4f} s")
    print(f"Pairs/sec:         {pair_count / elapsed:,.0f}")
    print(f"Connections:       {connections:,}")
    print(f"Matrix memory:     {adjacency.nbytes / 1024**2:.2f} MB")
    print("=" * 60)

    assert isinstance(adjacency, np.ndarray)
    assert adjacency.shape == (n_rows, n_rows)
    assert adjacency.dtype == np.float64

    assert np.all(np.diag(adjacency) == 0)
    assert np.array_equal(adjacency, adjacency.T)
