from typing import Any, List, Optional

from classification_model.processing.validation import TitanicDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "name": "Frédéric, Mr. D.",
                        "pclass": 1,
                        "age": 25.0,
                        "sibsp": 0,
                        "fare": 300,
                        "sex": "male",
                        "cabin": "E203",
                        "embarked": "S",
                    }
                ]
            }
        }
