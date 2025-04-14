# uvicorn run_LLama:app --reload
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import json

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Ragarenn
API_KEY = "sk-3530dcd8a156413598eb38745f523c3e"  # Remplacez par votre clé générée
BASE_URL = "https://ragarenn.eskemm-numerique.fr/equipe5"

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
Donne une note entre 0 à {req.max_grade}.
Donne des commentaires expliquant la note. Essaie de donner des commentaires aussi généraux et courts que possible qui puissent être appliqués à d'autres réponses similaires.
Sois juste dans votre notation.
Tiens compte du pas {req.step} lors de la notation
Donne ta réponse sous la forme de :
Note : X/{req.max_grade}
Titre du commentaire 1: ...
Commentaire 1: ... 
Titre du commentaire 2: ...
Commentaire 2: ...
Titre du commentaire n: ...
Commentaire n: ...
"""
    
    # Requête vers l'API Ragarenn
    url = BASE_URL + "/api/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "codestral:latest",  # Vous pouvez choisir un autre modèle disponible
        "messages": [
            {"role": "system", "content": "Tu es une assistante de notation précise et objective."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload  # Utilisez json au lieu de data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            return {"response": result["choices"][0]["message"]["content"]}
        else:
            return {"error": f"Erreur API: {response.status_code}", "details": response.text}
    
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}