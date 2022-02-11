import typing as t
from pathlib import Path

import joblib
import pandas as pd
from classification_model.pp_pipeline import pp_pipe
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def download_dataset() -> None:
    """Download the dataset from online source."""
    df = pd.read_csv(config.app_config.data_url)
    df.to_csv(Path(f"{DATASET_DIR}/{config.app_config.data_file}"), index=False)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # pre-process the data
    pp_data = pp_pipe.fit_transform(df, None)

    return pp_data


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()