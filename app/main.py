from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse 
from classifier import classify_prompt
import uvicorn
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))


# Mount the static files directory
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Define request model
class ClassificationRequest(BaseModel):
    prompt: str

# Define response model
class ClassificationResponse(BaseModel):
    prediction: str

@app.post("/classify")
async def classify(request: ClassificationRequest):
    # Extract the prompt from the request
    prompt = request.prompt

    # Perform classification using your classifier function
    prediction = classify_prompt(prompt)

    # Create the response
    response = ClassificationResponse(prediction=prediction)

    result_dict = {"classification": response.prediction}

    return result_dict

# Route for the root URL ("/") to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return open(os.path.join(current_dir, "static/index.html")).read()

# Run the FastAPI app using Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)