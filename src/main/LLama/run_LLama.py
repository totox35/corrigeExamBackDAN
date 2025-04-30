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
API_KEY = "sk-3530dcd8a156413598eb38745f523c3e"
BASE_URL = "https://ragarenn.eskemm-numerique.fr/equipe5"

class TextComment(BaseModel):
    """
    A model representing a text comment.

    Attributes:
        id (int): The id of the comment
        text (str): The text of the comment.
        description (str): The description of the comment.
        zonegeneratedid (str): The id of zone generated of the comment.
        questionid (int): The id of the question linked to the comment.
    """
    id: Optional[int] = None
    text: Optional[str] = None
    description: Optional[str] = None
    zonegeneratedid: Optional[str] = None
    question_id: Optional[int] = None

class GradedComment(BaseModel):
    """
    A model representing a graded comment.

    Attributes:
        id (int): The id of the comment
        text (str): The text of the comment.
        description (str): The description of the comment.
        zonegeneratedid (str): The id of zone generated of the comment.
        grade (float): The grade associated with the comment.
        questionid (int): The id of the question linked to the comment.
    """
    id: Optional[int] = None
    text: Optional[str] = None
    description: Optional[str] = None
    zonegeneratedid: Optional[str] = None
    grade: Optional[int] = None
    question_id: Optional[int] = None

class GradeRequest(BaseModel):
    """
    A model representing a request for grading a student's answer.

    Attributes:
        question (str): The question being graded.
        student_answer (str): The student's answer to the question.
        max_grade (int): The maximum grade possible for the question.
        step (float): The grading step or increment.
        existing_comments (Optional[List[TextComment]]): A list of existing text comments, if any.
        relevant_chunks (Optional[str]): Relevant chunks of information related to the question.
    """
    question: str
    # notes: str
    student_answer: List[str]
    max_grade: int
    step: float
    existing_comments: Optional[List[TextComment]] = Field(default_factory=list)
    relevant_chunks: List[str]

class GradeRequestGradedComment(BaseModel):
    """
    A model representing a request for grading a student's answer with graded comments.

    Attributes:
        question (str): The question being graded.
        student_answer (str): The student's answer to the question.
        max_grade (int): The maximum grade possible for the question.
        step (float): The grading step or increment.
        existing_comments (Optional[List[GradedComment]]): A list of existing graded comments, if any.
        grade_type (str): The type of grading (e.g., 'positive', 'negative').
        relevant_chunks (Optional[str]): Relevant chunks of information related to the question.
    """
    question: str
    # notes: str
    student_answer: List[str]
    max_grade: int
    step: float
    existing_comments: Optional[List[GradedComment]] = Field(default_factory=list)
    grade_type: str
    relevant_chunks: List[str]

class TCommentRequest(BaseModel):
    """
    A model representing a request for generating text comments.

    Attributes:
        question (str): The question being evaluated.
        student_answers (List[str]): A list of student answers to the question.
        nb_comments (int): The number of comments to generate.
        relevant_chunks (Optional[str]): Relevant chunks of information related to the question.
    """
    question: str
    # notes: str
    student_answers: List[str]
    nb_comments: int
    relevant_chunks: List[str]


class GCommentRequest(BaseModel):
    """
    A model representing a request for generating graded comments.

    Attributes:
        question (str): The question being evaluated.
        student_answers (List[str]): A list of student answers to the question.
        nb_comments (int): The number of comments to generate.
        grade_type (str): The type of grading (e.g., 'positive', 'negative').
        step (float): The grading step or increment.
        max_grade (int): The maximum grade possible for the question.
        relevant_chunks (Optional[str]): Relevant chunks of information related to the question.
    """
    question: str
    # notes: str
    student_answers: List[str]
    nb_comments: int
    grade_type: str
    step: float
    max_grade:int
    relevant_chunks: List[str]

@app.post("/api/grade_with_text_comments")
def grade(req: GradeRequest):
    """
    Evaluates a student's response by generating graded comments based on the provided context.

    Args:
        req (GradeRequest): The request data containing the question,
                             the student's answer, existing comments,
                             relevant context chunks, the maximum grade,
                             and the grading step.

    Returns:
        dict: A dictionary containing the generated response or an error message.
    """
    
    comments_text = ""
    if req.existing_comments and len(req.existing_comments) > 0:
        for i, comment in enumerate(req.existing_comments):
            comments_text += f"Titre du commentaire {i+1}: {comment.text}\n"
            comments_text += f"Commentaire {i+1}: {comment.description}\n"
    else:
        comments_text = "Aucun commentaire existant."

    context = ""
    if req.relevant_chunks and len(req.relevant_chunks) > 0:
        for i, chunk in enumerate(req.relevant_chunks):
            context += f"Chunk {i+1}: {chunk}\n"
    else:
        context = "Aucun contexte fourni."

    print(context)

    prompt = f"""
    Tu es une assistante pédagogique spécialisée dans la notation d'examens.

    Voici le contexte du cours: "{context}"
    QUESTION: "{req.question}"  
    RÉPONSES DES ÉTUDIANTS : "{req.student_answer}"  
    Commentaires existants :  
    {comments_text}

    Ta tâche :  
    - Attribue une note juste entre 0 et {req.max_grade}, en tenant compte du pas de notation ({req.step}).  
    - Fournis des commentaires courts et généraux, applicables aux plusieurs réponses plutards à partir du contexte fourni et de tes propres connaissances. 
    - Limite chaque commentaire à 3 à 5 mots. Évite les phrases complètes. 
    - Si des commentaires existants sont fournis, **utilise-les exactement comme ils sont**, même s’ils contiennent des erreurs, **à condition qu’ils soient pertinents**.  
    - Si aucun commentaire existant ne convient, génère-en de nouveaux.  

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
        response = requests.post(url, headers=headers,json=payload,  timeout=10)
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    
    except requests.exceptions.Timeout:
        print("Request timeout. Try again.")
    except requests.exceptions.RequestException as e:
        print(f"An error was raised : {e}")
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

@app.post("/api/grade_with_graded_comments")
def grade_graded_comment(req: GradeRequestGradedComment):
    """
    Evaluates a student's response by generating graded comments based on the provided context.

    Args:
        req (GradeRequestGradedComment): The request data containing the question,
                                        the student's answer, existing comments,
                                        relevant context chunks, the grade type,
                                        the grading step, and the maximum grade.

    Returns:
        dict: A dictionary containing the generated response or an error message.
    """

    
    comments_text = ""
    if req.existing_comments and len(req.existing_comments) > 0:
        for i, comment in enumerate(req.existing_comments):
            comments_text += f"Titre du commentaire {i+1}: {comment.text}\n"
            comments_text += f"Commentaire {i+1}: {comment.description}\n"
            comments_text += f"Note du commentaire{i+1}: {comment.grade}\n"
    else:
        comments_text = "Aucun commentaire existant"

    context = ""
    if req.relevant_chunks and len(req.relevant_chunks) > 0:
        for i, chunk in enumerate(req.relevant_chunks):
            context += f"Chunk {i+1}: {chunk}\n"
    else:
        context = "Aucun contexte fourni."

    prompt = f"""
    Tu es une assistante pédagogique spécialisée dans la notation d'examens.


    Contexte pour construire les commentaires :{context}
    Commentaires existants :  {comments_text}
    QUESTION : "{req.question}"  
    RÉPONSES DES ÉTUDIANTS : "{req.student_answer}"  

    Ta tâche :  
    - Génère un ou plusieurs commentaires expliquant l'évaluation de ces réponses à partir du contexte fourni et de tes propres connaissances.
    - Limite le nombre de commentaires au maximum, afin de garder un commentaire seulement s'il est pertinent.
    - Si les commentaires existants sont suffisants, ne propose pas d'autres commentaires à part les commentaires existants.  
    - Chaque commentaire doit être :
        • concis (3 à 5 mots),  
        • applicable à d'autres réponses similaires,  
        • cohérent avec la matière de la question,  
        • accompagné d'une **note {req.grade_type}**, en respectant le pas de notation de {req.step}. 
        • prends en compte que le maximum note pour cette question est {req.max_grade} donc enlever/ajouter plus que {req.max_grade}.  
    - Si des commentaires existants sont disponibles, **réutilise-les exactement (memes titres, memes commentaire et meme note)** (sans les modifier, même s’ils comportent des erreurs), si l’un d’eux correspond à ce que tu veux proposer.  
    - Sinon, crée de nouveaux commentaires.  
    - La note globale sera calculée automatiquement à partir des notes des commentaires, donc **ne donne pas de note finale globale**.

    ⚠️ Chaque note de commentaire doit être un multiple de {req.step} (exemples : 0, {req.step}, {req.step * 2}, etc.).
    ⚠️ N'utilise pas 0 comme note. Chaque commentaire doit avoir un impact minimal,tout en étant pertinent, sur la note finale.
    ⚠️ La somme des notes des commentaires que tu renvoies ne doit pas être inférieure à 0, et ne doit pas dépasser {req.max_grade}.

    ⚠️ Format strictement requis :

    Titre du commentaire 1 : ...  
    Commentaire 1 : ...  
    Note du commentaire 1 : ...

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
        response = requests.post(url, headers=headers,json=payload,  timeout=10)
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    
    except requests.exceptions.Timeout:
        print("Request timeout. Try again.")
    except requests.exceptions.RequestException as e:
        print(f"An error was raised : {e}")
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


@app.post("/api/propose_text_comments")
def propose_comments(req: TCommentRequest):
    """
    Generates a specified number of text comments for evaluating student responses.

    Args:
        req (TCommentRequest): The request data containing the question,
                                student answers, and the number of comments to generate.

    Returns:
        dict: A dictionary containing the generated comments or an error message.
    """

    context = ""
    if req.relevant_chunks and len(req.relevant_chunks) > 0:
        for i, chunk in enumerate(req.relevant_chunks):
            context += f"Chunk {i+1}: {chunk}\n"
    else:
        context = "Aucun contexte fourni."

    prompt = f"""
    Tu es une assistante pédagogique spécialisée dans l'évaluation d'examens.

    Contexte pour construire les commentaires :{context}
    QUESTION : "{req.question}"

    RÉPONSES DES ÉTUDIANTS : "{req.student_answers}"

    TÂCHE :
    - Génère exactement {req.nb_comments} commentaires courts et précis pour évaluer différentes réponses d'étudiants à partir du contexte fourni et de tes propres connaissances. 
    - Exactement {req.nb_comments} commentaires pas plus pas moins!
    - Chaque commentaire doit :
        • être concis (3 à 5 mots),
        • être réutilisable pour des réponses similaires,
        • couvrir un point fort ou une erreur fréquente,
        • être pertinent par rapport à la matière de la question.

    FORMAT EXACT REQUIS (respecte strictement ce format) :

    {'UNIQUEMENT:' if req.nb_comments == 1 else ''}
    Titre du commentaire 1 : [Titre bref et descriptif]  
    Commentaire 1 : [Commentaire court et précis]  

    {'' if req.nb_comments == 1 else '''[...]

    Titre du commentaire '''+str(req.nb_comments)+''' : [Titre bref et descriptif]  
    Commentaire '''+str(req.nb_comments)+''' : [Commentaire court et précis]'''}
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
        response = requests.post(url, headers=headers,json=payload,  timeout=10)
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    
    except requests.exceptions.Timeout:
        print("Request timeout. Try again.")
    except requests.exceptions.RequestException as e:
        print(f"An error was raised : {e}")
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


@app.post("/api/propose_graded_comments")
def propose_graded_comments(req: GCommentRequest):
    """
    Generates a specified number of graded comments for evaluating student responses.

    Args:
        req (GCommentRequest): The request data containing the question,
                                student answers, number of comments to generate,
                                grade type, grading step, and maximum grade.

    Returns:
        dict: A dictionary containing the generated comments or an error message.
    """

    context = ""
    if req.relevant_chunks and len(req.relevant_chunks) > 0:
        for i, chunk in enumerate(req.relevant_chunks):
            context += f"Chunk {i+1}: {chunk}\n"
    else:
        context = "Aucun contexte fourni."


    prompt = f"""
    Tu es une assistante pédagogique spécialisée dans l'évaluation d'examens.

    QUESTION : "{req.question}"

    RÉPONSES DES ÉTUDIANTS : "{req.student_answers}"

    TÂCHE :
    - Génère exactement {req.nb_comments} commentaires courts et précis pour évaluer différentes réponses d'étudiants à partir du contexte fourni et de tes propres connaissances.
    - Exactement {req.nb_comments} commentaires pas plus pas moins!
    - Chaque commentaire doit :
        • être concis (3 à 5 mots),
        • être adapté à des réponses similaires,
        • être en lien avec la matière de la question,
        • refléter un point {req.grade_type} ({'erreur ou manque' if req.grade_type == 'negative' else 'point fort ou réussite'}).

    - Chaque commentaire doit inclure une **note** ({req.grade_type}).
    ⚠️ Chaque note de commentaire doit être un multiple de {req.step} (exemples : 0, {req.step}, {req.step * 2}, etc.).Le maximum pour cette question est {req.max_grade}.
    N'utilise pas 0 comme note. Chaque commentaire doit avoir un impact sur la note finale.

    Contexte pour construire les commentaires :
    {context}

    FORMAT EXACT REQUIS (respecte strictement ce format) :

    {'UNIQUEMENT:' if req.nb_comments == 1 else ''}
    Titre du commentaire 1 : [Titre bref et descriptif]  
    Commentaire 1 : [Commentaire court et précis]  
    Note du commentaire 1 :

    {'' if req.nb_comments == 1 else '''[...]

    Titre du commentaire '''+str(req.nb_comments)+''' : [Titre bref et descriptif]  
    Commentaire '''+str(req.nb_comments)+''' : [Commentaire court et précis]  
    Note du commentaire '''+str(req.nb_comments)+''' :'''}
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
        response = requests.post(url, headers=headers,json=payload,  timeout=10)
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    
    except requests.exceptions.Timeout:
        print("Request timeout. Try again.")
    except requests.exceptions.RequestException as e:
        print(f"An error was raised : {e}")
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
