# python -m pytest tests/test_rust_integration.py tests/test_gnn_integration.py -v -s --tb=short 2>&1

import numpy as np
import pytest


class TestPrintHello:
    """Test the print_hello function from Rust module."""

    def test_print_hello(self):
        """Test that print_hello executes without raising an exception."""
        from likelihood import rust_py_integration

        rust_py_integration.print_hello()


class TestCalAdjacencyMatrix:
    """Test the cal_adjacency_matrix function from Rust module."""

    def test_basic_functionality(self):
        """Test basic adjacency matrix calculation with small dataset."""
        np.random.seed(42)
        data = np.random.rand(10, 5).astype(np.float64)

        from likelihood import rust_py_integration

        adj_matrix = rust_py_integration.cal_adjacency_matrix(
            data=data,
            n_cols=5,
            similarity=3,
            threshold=0.05,
        )

        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (10, 10)

        assert np.array_equal(
            adj_matrix,
            adj_matrix.T,
        ), "Adjacency matrix should be symmetric"

        assert np.all(np.diag(adj_matrix) == 0), "Adjacency matrix should have no self-loops"

        assert np.all(
            (adj_matrix == 0) | (adj_matrix == 1)
        ), "Adjacency matrix should contain only 0 and 1"

    def test_empty_dataset(self):
        """Test that an empty dataset raises an appropriate error."""
        from likelihood import rust_py_integration

        data = np.empty((0, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="at least one row"):
            rust_py_integration.cal_adjacency_matrix(
                data=data,
                n_cols=5,
                similarity=3,
                threshold=0.05,
            )

    def test_single_row(self):
        """Test with a single row dataset."""
        np.random.seed(42)
        data = np.random.rand(1, 5).astype(np.float64)

        from likelihood import rust_py_integration

        adj_matrix = rust_py_integration.cal_adjacency_matrix(
            data=data,
            n_cols=5,
            similarity=3,
            threshold=0.05,
        )

        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (1, 1)
        assert adj_matrix[0, 0] == 0

    def test_invalid_n_cols(self):
        """Test that an incorrect number of columns raises an error."""
        data = np.random.rand(10, 5).astype(np.float64)

        from likelihood import rust_py_integration

        with pytest.raises(ValueError, match="n_cols"):
            rust_py_integration.cal_adjacency_matrix(
                data=data,
                n_cols=4,
                similarity=3,
                threshold=0.05,
            )

    def test_zero_n_cols(self):
        """Test that n_cols=0 raises an error."""
        data = np.empty((10, 0), dtype=np.float64)

        from likelihood import rust_py_integration

        with pytest.raises(ValueError, match="n_cols"):
            rust_py_integration.cal_adjacency_matrix(
                data=data,
                n_cols=0,
                similarity=0,
                threshold=0.05,
            )

    def test_similarity_greater_than_n_cols(self):
        """Test that similarity cannot exceed the number of columns."""
        data = np.random.rand(10, 5).astype(np.float64)

        from likelihood import rust_py_integration

        with pytest.raises(ValueError, match="similarity"):
            rust_py_integration.cal_adjacency_matrix(
                data=data,
                n_cols=5,
                similarity=6,
                threshold=0.05,
            )

    def test_no_similar_rows(self):
        """Test that dissimilar rows produce no edges."""
        data = np.array(
            [
                [1.0, 1.0, 1.0],
                [10.0, 10.0, 10.0],
                [100.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        )

        from likelihood import rust_py_integration

        adj_matrix = rust_py_integration.cal_adjacency_matrix(
            data=data,
            n_cols=3,
            similarity=3,
            threshold=0.05,
        )

        expected = np.zeros((3, 3), dtype=np.float64)

        assert np.array_equal(adj_matrix, expected)


def test_rust_module_import():
    """Test that the Rust module can be imported."""
    from likelihood import rust_py_integration

    assert hasattr(
        rust_py_integration,
        "print_hello",
    )

    assert hasattr(
        rust_py_integration,
        "cal_adjacency_matrix",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
