from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model import config
from classification_model.processing.features import (
    ExtractLetterTransformer,
    ExtractTitleTransformer,
)

# Main Pipeline
titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.numerical_vars
            ),
        ),
        # ===== FEATURE ENGINEERING =====
        # Extract title from name variable
        (
            "extract_title",
            ExtractTitleTransformer(
                variable=config.model_config.variable_to_extract_title
            ),
        ),
        # Extract first letter from cabin
        (
            "extract_letter",
            ExtractLetterTransformer(
                variables=config.model_config.variable_to_get_cabin_goup
            ),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=config.model_config.rare_label_tol,
                variables=config.model_config.categorical_vars
                + config.model_config.feat_eng_cat_vars,
                n_categories=1,
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True,
                variables=(
                    config.model_config.categorical_vars
                    + config.model_config.feat_eng_cat_vars
                ),
            ),
        ),
        # == SCALING ======
        # scale using standardization
        ("scaler", StandardScaler()),
        # == MODEL ======
        # logistic regression (use C=0.0005 and random_state=0)
        (
            "Logit",
            LogisticRegression(
                C=config.model_config.C, random_state=config.model_config.random_state
            ),
        ),
    ]
)
