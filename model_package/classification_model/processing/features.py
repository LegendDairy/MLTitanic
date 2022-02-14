import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from classification_model import config


def get_first_cabin(row):
    """Retain only the first cabin if more than 1 are available per passenger"""
    try:
        return row.split()[0]
    except AttributeError:
        return np.nan


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].apply(get_first_cabin)
            X[feature] = X[feature].str[0]

        return X


def get_title(passenger):
    """"Helper function to extract title from a name."""
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


class ExtractTitleTransformer(BaseEstimator, TransformerMixin):
    # Extract title from name variable and removes name variable

    def __init__(self, variable):
        if not isinstance(variable, str):
            raise ValueError("Variable should be a string")

        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[config.model_config.new_title_variable] = X[self.variable].apply(get_title)
        X.drop(self.variable, axis=1, inplace=True)

        return X


class RemoveQuestionMarks(BaseEstimator, TransformerMixin):
    # Extract title from name variable and removes name variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.replace(config.model_config.missing_variable_id, np.nan)

        return X


class RecastVariables(BaseEstimator, TransformerMixin):
    # Recast strings to floats

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Cast numerical values to floats
        for var in config.model_config.variables_to_recast_to_flt:
            X[var] = X[var].astype("float")

        return X
