from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis system. "
                        "Return structured JSON only. "
                        "Sentiment must be one of: positive, negative, neutral. "
                        "Rating must be 1-5 where 5 is highly positive and 1 is highly negative."
                    ),
                },
                {
                    "role": "user",
                    "content": request.comment,
                },
            ],
            response_format=SentimentResponse,
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))