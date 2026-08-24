import pytest


# Fixtures for common test data
@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    import numpy as np

    np.random.seed(42)
    return np.random.rand(10, 5).astype(np.float64)


@pytest.fixture
def iris_data():
    """Load iris dataset for integration tests."""
    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df
