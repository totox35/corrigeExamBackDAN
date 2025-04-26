# uvicorn run_LLama:app --reload

# curl -X GET https://ragarenn.eskemm-numerique.fr/equipe5/api/models \   -H "Authorization: Bearer sk-3530dcd8a156413598eb38745f523c3e"

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

class GradedComment(BaseModel):
    id: Optional[int] = None
    text: Optional[str] = None
    description: Optional[str] = None
    zonegeneratedid: Optional[str] = None
    grade: Optional[int] = None
    question_id: Optional[int] = None

class GradeRequest(BaseModel):
    question: str
    # notes: str
    student_answer: str
    max_grade: int
    step: float
    existing_comments: Optional[List[TextComment]] = Field(default_factory=list)

class GradeRequestGradedComment(BaseModel):
    question: str
    # notes: str
    student_answer: str
    max_grade: int
    step: float
    existing_comments: Optional[List[GradedComment]] = Field(default_factory=list)
    grade_type: str

class TCommentRequest(BaseModel):
    question: str
    # notes: str
    student_answers: List[str]
    nb_comments: int


class GCommentRequest(BaseModel):
    question: str
    # notes: str
    student_answers: List[str]
    nb_comments: int
    grade_type: str
    step: float
    max_grade:int

@app.post("/api/grade_with_text_comments")
def grade(req: GradeRequest):
    
    comments_text = ""
    if req.existing_comments and len(req.existing_comments) > 0:
        for i, comment in enumerate(req.existing_comments):
            comments_text += f"Titre du commentaire {i+1}: {comment.text}\n"
            comments_text += f"Commentaire {i+1}: {comment.description}\n"
    else:
        comments_text = "Aucun commentaire existant."

    prompt = f"""
Tu es une assistante pédagogique spécialisée dans la notation d'examens.

Voici la question : "{req.question}"  
Voici la réponse de l'étudiant : "{req.student_answer}"  

Ta tâche :  
- Attribue une note juste entre 0 et {req.max_grade}, en tenant compte du pas de notation ({req.step}).  
- Fournis des commentaires courts et généraux, applicables à des réponses similaires. 
- Limite chaque commentaire à 3 à 10 mots. Évite les phrases complètes. 
- Si des commentaires existants sont fournis, **utilise-les exactement comme ils sont**, même s’ils contiennent des erreurs, **à condition qu’ils soient pertinents**.  
- Si aucun commentaire existant ne convient, génère-en de nouveaux.  

Commentaires existants :  
{comments_text}

⚠️ Format strictement requis pour ta réponse :

Note : X/{req.max_grade}  
Titre du commentaire 1 : ...  
Commentaire 1 : ...  
Titre du commentaire 2 : ...  
Commentaire 2 : ...  
[...]  
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

@app.post("/api/grade_with_graded_comments")
def grade(req: GradeRequestGradedComment):
    
    comments_text = ""
    if req.existing_comments and len(req.existing_comments) > 0:
        for i, comment in enumerate(req.existing_comments):
            comments_text += f"Titre du commentaire {i+1}: {comment.text}\n"
            comments_text += f"Commentaire {i+1}: {comment.description}\n"
            comments_text += f"Note du commentaire{i+1}: {comment.grade}\n"
    else:
        comments_text = "Aucun commentaire existant"

    prompt = f"""
Tu es une assistante pédagogique spécialisée dans la notation d'examens.

Voici la question : "{req.question}"  
Voici la réponse de l'étudiant : "{req.student_answer}"  

Ta tâche :  
- Génère des commentaires expliquant l'évaluation de cette réponse.  
- Chaque commentaire doit être :
    • concis (3 à 10 mots),  
    • applicable à d'autres réponses similaires,  
    • cohérent avec la matière de la question,  
    • accompagné d'une **note {req.grade_type}**, en respectant le pas de notation de {req.step} (le maximum pour cette question est {req.max_grade}).  
- Si des commentaires existants sont disponibles, **réutilise-les exactement (memes titres, memes commentaire et meme note)** (sans les modifier, même s’ils comportent des erreurs), si l’un d’eux correspond à ce que tu veux proposer.  
- Sinon, crée de nouveaux commentaires.  
- La note globale sera calculée automatiquement à partir des notes des commentaires, donc **ne donne pas de note finale globale**.

Commentaires existants :  
{comments_text}

⚠️ Chaque note de commentaire doit être un multiple de {req.step} (exemples : 0, {req.step}, {req.step * 2}, etc.).N'utilise pas 0 comme note. Chaque commentaire doit avoir un impact sur la note finale.

⚠️ Format strictement requis :

Titre du commentaire 1 : ...  
Commentaire 1 : ...  
Note du commentaire 1 : ...

Titre du commentaire 2 : ...  
Commentaire 2 : ...  
Note du commentaire 2 : ...

[...]

Titre du commentaire n : ...  
Commentaire n : ...  
Note du commentaire n : ...
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


@app.post("/api/propose_text_comments")
def propose_comments(req: TCommentRequest):

    prompt = f"""
Tu es une assistante pédagogique spécialisée dans l'évaluation d'examens.

QUESTION : "{req.question}"

RÉPONSES DES ÉTUDIANTS : "{req.student_answers}"

TÂCHE :
- Génère exactement {req.nb_comments} commentaires courts et précis pour évaluer différentes réponses d'étudiants. 
- Exactement {req.nb_comments} commentaires pas plus pas moins!
- Chaque commentaire doit :
    • être concis (3 à 10 mots),
    • être réutilisable pour des réponses similaires,
    • couvrir un point fort ou une erreur fréquente,
    • être pertinent par rapport à la matière de la question.

FORMAT EXACT REQUIS (respecte strictement ce format) :

Titre du commentaire 1 : [Titre bref et descriptif]  
Commentaire 1 : [Commentaire court et précis]

Titre du commentaire 2 : [Titre bref et descriptif]  
Commentaire 2 : [Commentaire court et précis]

[...]

Titre du commentaire {req.nb_comments} : [Titre bref et descriptif]  
Commentaire {req.nb_comments} : [Commentaire court et précis]
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


@app.post("/api/propose_graded_comments")
def propose_comments(req: GCommentRequest):

    prompt = f"""
Tu es une assistante pédagogique spécialisée dans l'évaluation d'examens.

QUESTION : "{req.question}"

RÉPONSES DES ÉTUDIANTS : "{req.student_answers}"

TÂCHE :
- Génère exactement {req.nb_comments} commentaires courts et précis pour évaluer différentes réponses d'étudiants.
- Exactement {req.nb_comments} commentaires pas plus pas moins!
- Chaque commentaire doit :
    • être concis (3 à 10 mots),
    • être adapté à des réponses similaires,
    • être en lien avec la matière de la question,
    • refléter un point {req.grade_type} ({'erreur ou manque' if req.grade_type == 'negative' else 'point fort ou réussite'}).

- Chaque commentaire doit inclure une **note** ({req.grade_type}).
⚠️ Chaque note de commentaire doit être un multiple de {req.step} (exemples : 0, {req.step}, {req.step * 2}, etc.).Le maximum pour cette question est {req.max_grade}.
N'utilise pas 0 comme note. Chaque commentaire doit avoir un impact sur la note finale.

FORMAT EXACT REQUIS (respecte strictement ce format) :

Titre du commentaire 1 : [Titre bref et descriptif]  
Commentaire 1 : [Commentaire court et précis]  
Note du commentaire 1 :

Titre du commentaire 2 : [Titre bref et descriptif]  
Commentaire 2 : [Commentaire court et précis]  
Note du commentaire 2 :

[...]

Titre du commentaire {req.nb_comments} : [Titre bref et descriptif]  
Commentaire {req.nb_comments} : [Commentaire court et précis]  
Note du commentaire {req.nb_comments} :
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
