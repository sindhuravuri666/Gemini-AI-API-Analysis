import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from textblob import TextBlob
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI()

# Load Gemini API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Use a Gemini model
model = genai.GenerativeModel("models/gemini-flash-latest")

@app.get("/")
async def root():
    return {"message": "Gemini FastAPI server is running"}


def get_ai_response(user_query: str):
    try:
        response = model.generate_content(user_query)
        return response.text.strip()
    except Exception as e:
        return f"Failed to connect to AI server: {str(e)}"


@app.post("/get_ai_response")
async def ai_response(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return JSONResponse(content={"message": "Query is empty"}, status_code=400)

    ai_response = get_ai_response(user_query)

    if ai_response.startswith("Failed to connect to AI server:"):
        return JSONResponse(content={"message": ai_response}, status_code=500)

    return JSONResponse(content={"response": ai_response})


@app.post("/analyze")
async def analyze_query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return JSONResponse(content={"message": "Query is empty"}, status_code=400)

    ai_response = get_ai_response(user_query)

    if ai_response.startswith("Failed to connect to AI server:"):
        return JSONResponse(content={"message": ai_response}, status_code=500)

    sentiment_score = TextBlob(ai_response).sentiment.polarity

    return JSONResponse(content={
        "query": user_query,
        "response": ai_response,
        "sentiment": sentiment_score
    })