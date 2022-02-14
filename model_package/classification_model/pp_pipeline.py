from sklearn.pipeline import Pipeline

from classification_model import config
from classification_model.processing.features import (
    RecastVariables,
    RemoveQuestionMarks,
)

# Pre-processing Pipeline
pp_pipe = Pipeline(
    [
        # replace question marks with np.nan
        ("remove_question_marks", RemoveQuestionMarks()),
        # recast strings to floats
        (
            "recast_variables",
            RecastVariables(variables=config.model_config.variables_to_recast_to_flt),
        ),
    ]
)
