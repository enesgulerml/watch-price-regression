from pydantic import BaseModel
from typing import Literal


class WatchFeatures(BaseModel):
    """
    The input schema for our API.
    Just like config.yaml, it defines all our features.
    """

    # -- Numerical Properties --
    Case_Diameter: float
    Water_Resistance: int
    Warranty_Years: int
    Weight_g: float

    # -- Categorical Features --
    Brand: str
    Gender: Literal['Male', 'Female', 'Unisex']
    Case_Color: str
    Glass_Shape: str
    Origin: str
    Case_Material: str
    Additional_Feature: str
    Strap_Color: str
    Strap_Material: str
    Mechanism: str
    Glass_Type: str
    Dial_Color: str

    class Config:
        json_schema_extra = {
            "example": {
                "Case_Diameter": 40.5,
                "Water_Resistance": 10,
                "Warranty_Years": 2,
                "Weight_g": 85.0,
                "Brand": "Seiko",
                "Gender": "Male",
                "Case_Color": "Silver",
                "Glass_Shape": "Flat",
                "Origin": "Japan",
                "Case_Material": "Steel",
                "Additional_Feature": "Luminous",
                "Strap_Color": "Black",
                "Strap_Material": "Leather",
                "Mechanism": "Automatic",
                "Glass_Type": "Sapphire",
                "Dial_Color": "Blue"
            }
        }