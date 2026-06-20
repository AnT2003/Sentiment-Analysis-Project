from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    comment: str = Field(..., title="User Comment", description="The text of the comment to analyze.")
    
class PredictResponse(BaseModel):
    sentiment: str = Field(..., title="Sentiment", description="The predicted sentiment: positive, neutral, or negative.")
    confidence: float = Field(None, title="Confidence Score", description="Probability of the predicted class.")
