"""Keras persistence helpers that follow the installed TensorFlow version.

TensorFlow > 2.15 (Keras 3) writes a single ``.keras`` archive and rejects the
``save_format`` argument, while TensorFlow <= 2.15 (Keras 2) expects
``save_format="tf"`` SavedModel directories.  ``EnsembleClassifier`` already
switches on the version inline; these helpers centralise the same rule for the
other custom models (``AutoClassifier``, ``GANRegressor``).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf
from packaging import version

#: First TensorFlow release where Keras 3 (and the ``.keras`` archive) is the default.
KERAS_ARCHIVE_VERSION = "2.15.0"


def keras_uses_archive() -> bool:
    """Whether the installed TensorFlow expects the native ``.keras`` archive."""
    return version.parse(tf.__version__) > version.parse(KERAS_ARCHIVE_VERSION)


def resolve_artifact(path: str | Path) -> Path:
    """Return the stored artifact, accepting ``name`` for a ``name.keras`` file."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    archive = candidate.with_name(candidate.name + ".keras")
    return archive if archive.exists() else candidate


def save_keras_model(model: Any, path: str | Path, *, archive: Optional[bool] = None) -> Path:
    """Persist ``model`` using the format supported by the installed stack.

    Parameters
    ----------
    model : Any
        Keras model to serialize.
    path : str or pathlib.Path
        Destination. On TensorFlow > 2.15 a ``.keras`` suffix is appended when
        missing; older releases write a SavedModel directory at ``path``.
    archive : bool, optional
        Force one of the two formats (used by the tests); by default the
        installed TensorFlow version decides.

    Returns
    -------
    pathlib.Path
        File or directory that was written.
    """
    target = Path(path)
    use_archive = keras_uses_archive() if archive is None else archive
    if use_archive:
        if target.suffix != ".keras":
            target = target.with_name(target.name + ".keras")
        model.save(str(target))
        return target
    model.save(str(target), save_format="tf")
    return target


def load_keras_model(
    path: str | Path,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = False,
) -> Any:
    """Load a ``.keras`` archive or a SavedModel directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Stored model. ``name`` is accepted when the artifact is ``name.keras``.
    custom_objects : dict, optional
        Classes Keras must know in order to rebuild the model.
    compile : bool, default False
        Whether to restore the training configuration saved with the model.

    Returns
    -------
    Any
        The deserialized Keras model.
    """
    artifact = resolve_artifact(path)
    return tf.keras.models.load_model(
        str(artifact), custom_objects=custom_objects or {}, compile=compile
    )
