import json
from typing import Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from app.config import settings
from app import __version__, schemas
from classification_model import __version__ as model_version
from classification_model.predict import make_prediction

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Preform health check of the API
    """

    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleTitanicDataInputs) -> Any:
    """
    Make Titanic survival predictions with the classification model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # test for errors
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    # log predictions
    logger.info(f"Prediction results: {results.get('predictions').tolist()}")

    # results are a numpy array so convert to list for validation
    results["predictions"] = results.get('predictions').tolist()

    return results
