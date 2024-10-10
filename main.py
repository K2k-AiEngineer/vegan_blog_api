# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI application
app = FastAPI()

# Allow CORS for all origins (adjust as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Zero-Shot Classification pipeline
classifier = pipeline("zero-shot-classification")

# Define a request model
class BlogPost(BaseModel):
    text: str

# Function to classify the text
def classify_text(blog_post: str):
    labels = ["vegan", "non-vegan"]
    result = classifier(blog_post, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]

# Define the API endpoint for classification
@app.post("/classify/")
async def classify_blog(blog_post: BlogPost):
    try:
        classification, score = classify_text(blog_post.text)
        return {
            "classification": classification,
            "confidence_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application with `uvicorn main:app --reload`
