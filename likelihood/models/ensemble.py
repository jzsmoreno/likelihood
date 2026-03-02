import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from packaging import version
from sklearn.base import BaseEstimator, ClassifierMixin
import json


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble of AutoClassifier models with different hyperparameters and random seeds.

    Parameters
    ----------
    n_models : `int`
        Number of models in the ensemble. Default is `5`.

    base_model_class : `class`
        The classifier model class to use (e.g., `AutoClassifier`). Default is `AutoClassifier`.

    param_ranges : `dict`, `optional`
        Dictionary specifying ranges for hyperparameter variation:
        - `units`: list of integers or tuple `(min, max)`
        - `activation`: list of activation function names
        - `num_layers`: list of integers or tuple `(min, max)`
        - `dropout`: list of floats or tuple `(min, max)`
        - `l2_reg`: list of floats or tuple `(min, max)`
        If `None`, default values are used.

    seed_range : `tuple`, `optional`
        Range for random seeds as (start, end). Default is `(0, 100)`.

    voting_method : `str`, `optional`
        Method to combine predictions: `soft` (average probabilities) or `hard` (majority vote). Default is `soft`.

    verbose : `int`, `optional`
        Verbosity level. `0` = silent, `1` = progress bar. Default is `0`.

    Attributes
    ----------
    models_ : `list`
        List of trained AutoClassifier instances.

    model_params_ : `list`
        List of parameter dictionaries used for each model.

    n_models_ : `int`
        Actual number of models created and trained.

    Methods
    -------
    `fit(X, y)`
        Train all ensemble members on the provided data.

    `predict(X)`
        Return predicted class labels using the voting method.

    `predict_proba(X)`
        Return probability estimates for each class (soft voting only).

    `get_model_scores()`
        Retrieve individual model performance metrics if available.

    Examples
    --------
    >>> ensemble = EnsembleClassifier(
    ...     n_models=5,
    ...     param_ranges={'units': (10, 20), 'activation': ['selu', 'relu']},
    ...     voting_method='soft'
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        n_models: int = 5,
        base_model_class=None,
        param_ranges: Optional[Dict] = None,
        seed_range: tuple = (0, 100),
        voting_method: str = "soft",
        verbose: int = 0,
    ):
        if base_model_class is None:
            from likelihood.models.deep import AutoClassifier

            base_model_class = AutoClassifier

        self.n_models = n_models
        self.base_model_class = base_model_class
        self.param_ranges = param_ranges or {}
        self.seed_range = seed_range
        self.voting_method = voting_method
        self.verbose = verbose

        # Initialize internal attributes
        self.scores = []
        self.configs = []
        self.models_ = []
        self.model_params_ = []
        self.all_history = []
        self.n_models_ = 0

    def _generate_model_configs(self) -> List[Dict]:
        """Generate unique configurations for each model in the ensemble."""
        self.configs = []
        default_ranges = {
            "units": [17],
            "activation": ["selu"],
            "num_layers": [1],
            "dropout": [None],
            "l2_reg": [0.0],
        }

        param_ranges = self.param_ranges or default_ranges

        def sample_param(name, integer=False):
            values = param_ranges.get(name, default_ranges[name])
            if isinstance(values, tuple):
                low, high = values
                if integer:
                    return np.random.randint(low, high + 1)
                else:
                    return np.random.uniform(low, high)
            return np.random.choice(values)

        for _ in range(self.n_models):
            config = {
                "units": sample_param("units", integer=True),
                "activation": sample_param("activation"),
                "num_layers": sample_param("num_layers", integer=True),
                "dropout": sample_param("dropout"),
                "l2_reg": sample_param("l2_reg"),
                "seed": np.random.randint(*self.seed_range),
            }
            self.configs.append(config)

        return self.configs

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """Train all models in the ensemble."""
        self.models_ = []
        self.all_history = []
        self.model_params_ = []
        configs = self.configs or self._generate_model_configs()

        for i, config in enumerate(configs):
            if self.verbose > 0 and (i + 1) % max(1, len(configs) // 5) == 0:
                print(f"Training model {i+1}/{len(configs)}...")

            tf.random.set_seed(config["seed"])
            np.random.seed(config["seed"])

            try:
                model = self.base_model_class(
                    input_shape_parm=X.shape[1],
                    num_classes=y.shape[1] if len(y.shape) > 1 else y.max() + 1,
                    units=config["units"],
                    activation=config["activation"],
                    l2_reg=config["l2_reg"],
                    num_layers=config["num_layers"],
                    dropout=config["dropout"],
                )

                model.compile(
                    optimizer="adam",
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.F1Score(threshold=0.5, average="micro")],
                )

                history = model.fit(X, y, verbose=0, **fit_kwargs)
                self.models_.append(model)
                self.all_history.append(history)
                self.model_params_.append(config)

            except Exception as e:
                print(f"Warning: Model {i+1} failed to train. Error: {e}")

        self.n_models_ = len(self.models_)
        if self.verbose > 0:
            print(f"Ensemble trained with {self.n_models_} models.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class (soft voting)."""
        all_probs = []
        for model in self.models_:
            probs = model.predict(X, verbose=0)
            if len(probs.shape) == 1:
                probs = tf.nn.softmax(probs).numpy()
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        if self.voting_method == "soft":
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
        elif self.voting_method == "hard":
            all_preds = []
            for model in self.models_:
                preds = model.predict(X, verbose=0)
                if len(preds.shape) > 1:
                    preds = np.argmax(preds, axis=1)
                else:
                    preds = (preds >= 0.5).astype(int)
                all_preds.append(preds.reshape(-1, 1))

            stacked = np.hstack(all_preds)
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=stacked)
        else:
            raise ValueError("voting_method must be 'soft' or 'hard'.")

    def get_model_scores(self):
        """Return performance metrics for each model (if available)."""
        if hasattr(self, "all_history"):
            self.scores = []
            for i, model in enumerate(self.models_):
                if hasattr(model, "history"):
                    loss = self.all_history[i].history.get("loss", [None])[0]
                    val_loss = self.all_history[i].history.get("val_loss", [None])[0]
                    f1 = self.all_history[i].history.get("f1_score", [None])[0]
                    val_f1 = self.all_history[i].history.get("val_f1_score", [None])[0]
                    self.scores.append(
                        {
                            "model_id": i + 1,
                            "params": self.model_params_[i],
                            "loss": loss,
                            "val_loss": val_loss,
                            "f1_score": f1,
                            "val_f1_score": val_f1,
                        }
                    )
        return self.scores

    def _save_keras_model(self, model, path: str) -> None:
        """
        Internal helper to save a Keras model depending on TF version.
        """
        is_updated = version.parse(tf.__version__) > version.parse("2.15.0")

        if is_updated:
            model.save(f"{path}")
        else:
            model.save(path, save_format="tf")

    def save(self, filepath: str) -> None:
        """
        Save ensemble metadata and models inside a directory
        named after the last part of filepath.
        """

        filepath = filepath.rstrip(".pkl")
        parent_dir = os.path.dirname(filepath)
        folder_name = os.path.basename(filepath)
        save_dir = os.path.join(parent_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        self.get_model_scores()

        save_dict = {
            "n_models": self.n_models,
            "param_ranges": self.param_ranges,
            "seed_range": self.seed_range,
            "voting_method": self.voting_method,
            "verbose": self.verbose,
            "configs": self.configs,
            "model_params_": self.model_params_,
            "scores": self.scores,
            "n_models_": self.n_models_,
        }

        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(save_dict, f, indent=4)

        for i, model in enumerate(self.models_):
            self._save_keras_model(model, os.path.join(save_dir, f"model_{i}"))

    @classmethod
    def load(cls, filepath: str, base_model_class=None):

        filepath = filepath.rstrip(".pkl")

        parent_dir = os.path.dirname(filepath)
        folder_name = os.path.basename(filepath)
        save_dir = os.path.join(parent_dir, folder_name)

        with open(os.path.join(save_dir, "meta.json"), "r") as f:
            save_dict = json.load(f)

        ensemble = cls.__new__(cls)

        for key, value in save_dict.items():
            setattr(ensemble, key, value)

        ensemble.models_ = []
        if base_model_class is None:
            from likelihood.models.deep import AutoClassifier

            base_model_class = AutoClassifier

        def _load_keras_model(path: str) -> None:
            """
            Internal helper to load a Keras model depending on TF version.
            """
            is_updated = version.parse(tf.__version__) > version.parse("2.15.0")

            if is_updated:
                return base_model_class.load(path)
            else:
                return tf.keras.models.load_model(path)

        for i in range(ensemble.n_models_):
            model_path = os.path.join(save_dir, f"model_{i}")
            model = _load_keras_model(model_path)
            ensemble.models_.append(model)

        ensemble.base_model_class = base_model_class

        return ensemble

    def __repr__(self):
        return (
            f"EnsembleClassifier(n_models={self.n_models}, "
            f"voting_method='{self.voting_method}', "
            f"trained_models={self.n_models_})"
        )
