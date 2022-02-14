from classification_model import config
from classification_model.pp_pipeline import pp_pipe
from classification_model.processing.features import (
    ExtractLetterTransformer,
    ExtractTitleTransformer,
)


def test_feature_engineering_transformers(sample_input_data):
    pp_data = pp_pipe.fit_transform(sample_input_data, None)
    cabin_extractor = ExtractLetterTransformer(
        variables=config.model_config.variable_to_get_cabin_goup
    )
    title_extractor = ExtractTitleTransformer(
        variable=config.model_config.variable_to_extract_title
    )

    assert sample_input_data["cabin"].iat[6] == "E12"

    assert sample_input_data["name"].iat[6] == "Anderson, Mr. Harry"

    transformed_data = cabin_extractor.fit_transform(pp_data)
    assert transformed_data["cabin"].iat[6] == "E"

    transformed_data = title_extractor.fit_transform(pp_data)
    assert transformed_data["title"].iat[6] == "Mr"
