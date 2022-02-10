import re

from regression_model import config


class ExtractLetterTransformer:
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('Variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


def get_title(passenger):
    """"Helper function to extract title from a name."""
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


class ExtractTitleTransformer:
    # Extract title from name variable and removes name variable

    def __init__(self, variable):
        if not isinstance(variable, str):
            raise ValueError('Variable should be a string')

        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[config.model_config.new_title_variable] = X[self.variable].apply(get_title)
        X.drop(self.variable, axis=1, inplace=True)

        return X