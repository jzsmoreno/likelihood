# python -m pytest tests/test_rust_integration.py tests/test_gnn_integration.py -v -s --tb=short 2>&1

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris


class TestVanillaGNNIntegration:
    """Test integration of Rust-calculated adjacency matrix with VanillaGNN."""

    def test_basic_gnn_flow(self):
        """Test that VanillaGNN works with Rust-calculated adjacency matrix."""
        np.random.seed(42)

        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names,
        )
        df["species"] = iris.target

        from likelihood import rust_py_integration

        data_array = df.iloc[:, :4].values.astype(np.float64)

        adj_matrix = rust_py_integration.cal_adjacency_matrix(
            data=data_array,
            n_cols=4,
            similarity=4,
            threshold=0.05,
        )

        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (len(df), len(df))
        assert np.array_equal(adj_matrix, adj_matrix.T)
        assert np.all(np.diag(adj_matrix) == 0)

        print(f"Calculated adjacency matrix shape: {adj_matrix.shape}")
        print(f"Number of connections: {np.sum(adj_matrix)}")

    def test_full_pipeline(self):
        """Test the full VanillaGNN training pipeline."""

        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names,
        )
        df["species"] = iris.target

        from likelihood import rust_py_integration

        data_array = df.iloc[:, :4].values.astype(np.float64)

        adj_matrix = rust_py_integration.cal_adjacency_matrix(
            data=data_array,
            n_cols=4,
            similarity=4,
            threshold=0.05,
        )

        assert adj_matrix.shape == (len(df), len(df))

        from likelihood.graph import Data
        from likelihood.graph.nn import VanillaGNN

        data = Data(df, "species")

        assert data.x.shape == (150, 4)
        assert data.adjacency is not None

        model = VanillaGNN(
            dim_in=data.x.shape[1],
            dim_h=8,
            dim_out=len(df["species"].unique()),
            rank=4,
        )

        assert model is not None

        model.fit(data, epochs=400, batch_size=32, test_size=0.3)
        output = model.predict(data)

        assert model.built
        assert output is not None

        print(f"Data x shape: {data.x.shape}")
        print(f"Data adjacency info: {type(data.adjacency)}")
        print("Model created and built successfully!")

        print("Model summary:")
        model.summary()

        # Predictions on the current dataset
        predicted_classes = model.predict(data)

        print("Predicted classes:", predicted_classes)
        print("Actual classes:", data.y)

        # Show prediction vs actual results
        df_results = pd.DataFrame(
            {
                "predicted": predicted_classes,
                "actual": data.y,
            }
        )

        print("Prediction results:")
        print(df_results.head(20))

        print("Prediction accuracy:")
        accuracy = np.mean(predicted_classes == data.y)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1: {model.test(data)}")


def test_adjacency_matrix_properties():
    """Test properties of the adjacency matrix from Rust."""

    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )

    from likelihood import rust_py_integration

    adj_matrix = rust_py_integration.cal_adjacency_matrix(
        data=data,
        n_cols=5,
        similarity=5,
        threshold=0.05,
    )

    assert isinstance(adj_matrix, np.ndarray)
    assert adj_matrix.shape == (6, 6)

    assert adj_matrix[0, 2] == 1
    assert adj_matrix[0, 3] == 1
    assert adj_matrix[0, 5] == 1
    assert adj_matrix[2, 3] == 1
    assert adj_matrix[2, 5] == 1
    assert adj_matrix[3, 5] == 1

    assert adj_matrix[0, 1] == 0
    assert adj_matrix[0, 4] == 0
    assert adj_matrix[1, 4] == 0

    assert np.array_equal(adj_matrix, adj_matrix.T)
    assert np.all(np.diag(adj_matrix) == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
