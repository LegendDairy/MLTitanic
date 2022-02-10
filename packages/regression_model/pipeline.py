from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, AddMissingIndicator, MeanMedianImputer
from regression_model import config
from regression_model.processing.features import ExtractLetterTransformer, ExtractTitleTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # Extract title from name variable
    ('extract_letter', ExtractTitleTransformer(variable=config.model_config.variable_to_extract_title)),


    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(imputation_method='missing',
                                                  variables=config.model_config.categorical_vars)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.model_config.numerical_vars_vars)),


    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='median',
                                            variables=config.model_config.numerical_vars_vars)),

    # Extract first letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.model_config.variable_to_get_cabin_goup)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(tol=config.model_config.rare_label_tol,
                                            variables=config.model_config.numerical_vars_vars,
                                            n_categories=1)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config.model_config.numerical_vars_vars)),


    # scale using standardization
    ('scaler', StandardScaler()),


    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=config.model_config.C, random_state=config.model_config.random_state)),
])
