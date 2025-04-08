# uvicorn run_LLama:app --reload

from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Add this after "app = FastAPI()"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Here is the relevant part of the class notes: "{req.notes}"

class GradeRequest(BaseModel):
    question: str
    # notes: str
    student_answer: str
    max_grade: int
    step: float

@app.post("/api/grade")
def grade(req: GradeRequest):
    prompt = f"""
Tu es une assistante de notation très utile.
Voici la question : "{req.question}"
Voici la réponse de l'étudiant : "{req.student_answer}"

Donne une note entre 0 à {req.max_grade} et un bref commentaire expliquant la note. Sois juste dans votre notation.
Donne ta réponse sous la forme de :
Note : X/{req.max_grade} 
Tiens compte du pas {req.step} lors de la notation
Commentaire : ...
"""

    response = requests.post(
        # "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        headers={"Authorization": f"Bearer Your password"},
        json={"inputs": prompt}
    )

    return response.json()
