from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
from pathlib import Path

# reroute to /src folder as that is where model and recommender file are stored
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from recommender import get_recommendations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    os.makedirs("../data/uploads", exist_ok=True)

    upload_path = f"../data/uploads/{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    recommendations = get_recommendations(upload_path)

    return {
        "filename": file.filename,
        "message": "File received successfully!",
        "recommendations": recommendations
    }