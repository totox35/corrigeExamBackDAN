

import requests
import json

# Configuration
API_KEY = "sk-3530dcd8a156413598eb38745f523c3e"  # Remplacez par votre clé générée
BASE_URL = "https://ragarenn.eskemm-numerique.fr/equipe5"  # URL de base sans le chemin spécifique


def envoyer_prompt(prompt, model_id="codestral:latest"):  # Utilisation du premier modèle disponible
    url = BASE_URL + "/api/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant IA utile et concis."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        return f"Erreur: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"Erreur de requête: {str(e)}"

# Test avec un prompt simple
if __name__ == "__main__":
    prompt_utilisateur = "Donnez-moi trois idées de projets innovants utilisant l'IA."
    
    # Essayons avec le premier modèle disponible
    model = "codestral:latest"
    print(f"Envoi de la requête avec le modèle {model}...")
    reponse = envoyer_prompt(prompt_utilisateur, model)
    print("\nRéponse:")
    print(reponse)