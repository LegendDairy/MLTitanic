from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def get_first_cabin(row):
    """Retain only the first cabin if more than 1 are available per passenger"""
    try:
        return row.split()[0]
    except:
        return np.nan


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()

    # Replace '?' with np.nan
    validated_data = validated_data.replace(config.model_config.missing_variable_id, np.nan)


    # Clean cabin feature
    validated_data[config.model_config.variable_to_get_cabin_goup] = \
        validated_data[config.model_config.variable_to_get_cabin_goup].apply(get_first_cabin)

    # Cast numerical values to floats
    for var in config.model_config.variables_to_recast_to_flt:
        validated_data[var] = validated_data[var].astype('float')

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    name: Optional[str]
    pclass: Optional[int]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    fare: Optional[float]
    sex: Optional[str]
    cabin: Optional[str]
    embarked: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
