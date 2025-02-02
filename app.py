from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from textblob import TextBlob
import requests

app = FastAPI()

# AI response function (calls external AI API like Gemini or OpenAI)

def get_ai_response(user_query: str):
    # Example for OpenAI API (replace it with Gemini or other APIs)
    api_key = "AIzaSyBTmqit2Oct0siWKTlMRM137Mev7_aSAwk"  # Replace with your actual API key
    url = "https://api.openai.com/v1/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "prompt": user_query,
        "max_tokens": 100,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['text'].strip()
    else:
        return "Failed to connect to AI server."

@app.post("/get_ai_response")
async def ai_response(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return JSONResponse(content={"message": "Query is empty"}, status_code=400)

    ai_response = get_ai_response(user_query)

    if ai_response == "Failed to connect to AI server.":
        return JSONResponse(content={"message": ai_response}, status_code=500)

    return JSONResponse(content={"response": ai_response})



@app.post("/analyze")
async def analyze_query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return JSONResponse(content={"message": "Query is empty"}, status_code=400)

    ai_response = get_ai_response(user_query)

    if ai_response == "Failed to connect to AI server.":
        return JSONResponse(content={"message": ai_response}, status_code=500)

    sentiment_score = TextBlob(ai_response).sentiment.polarity

    return JSONResponse(content={
        "query": user_query,
        "response": ai_response,
        "sentiment": sentiment_score
    })
    
