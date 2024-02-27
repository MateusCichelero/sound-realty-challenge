from pydantic import BaseModel, Field

class InputAll(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    waterfront: int = Field(..., ge=0)
    view: int = Field(..., ge=0)
    condition: int = Field(..., ge=0)
    grade: int = Field(..., ge=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    yr_built: int = Field(..., ge=0)
    yr_renovated: int = Field(..., ge=0)
    zipcode: str
    lat: float 
    long: float 
    sqft_living15: int = Field(..., ge=0)
    sqft_lot15: int = Field(..., ge=0)

class InputMinimal(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    zipcode: str


class InferenceOutput(BaseModel):
    prediction: float