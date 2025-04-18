# uvicorn run_LLama:app --reload

# curl -X GET https://ragarenn.eskemm-numerique.fr/equipe5/api/models \
#   -H "Authorization: Bearer sk-3530dcd8a156413598eb38745f523c3e"

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Union
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

class TextComment(BaseModel):
    id: Optional[int] = None
    text: Optional[str] = None
    description: Optional[str] = None
    zonegeneratedid: Optional[str] = None
    question_id: Optional[int] = None

class GradeRequest(BaseModel):
    question: str
    # notes: str
    student_answer: str
    max_grade: int
    step: float
    existing_comments: Optional[List[TextComment]] = Field(default_factory=list)

@app.post("/api/grade")
def grade(req: GradeRequest):
    
    comments_text = ""
    if req.existing_comments and len(req.existing_comments) > 0:
        for i, comment in enumerate(req.existing_comments):
            comments_text += f"Titre du commentaire {i+1}: {comment.text}\n"
            comments_text += f"Commentaire {i+1}: {comment.description}\n"
    else:
        comments_text = "Aucun commentaire existant"

    prompt = f"""
Tu es une assistante de notation très utile.
Voici la question : "{req.question}"
Voici la réponse de l'étudiant : "{req.student_answer}"
Donne une note entre 0 à {req.max_grade}.
Donne des commentaires expliquant la note. 
Si il y a deja des commentaires aui existe tu peux choisir entre ces commentaires et retourner la commentaire exactement  sans le changer meme s'il y a des erreurs grammatical dans les commentaires, les commentaires existants: {comments_text}. 
Essaie de donner des commentaires aussi généraux et courts que possible qui puissent être appliqués à d'autres réponses similaires. 
Essaie de choisir le commentaire deja existant,s'il y a un qui est assez proche que tu veux proposer. 
Oublie pas de donner un titre pour chaque commentaire!
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
        # "model": "codestral:latest",  
        "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
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