import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Llink API Server")

# 1. CONFIGURATION SÉCURISÉE
# Le token doit être ajouté dans le "Dashboard Render" -> Env Vars sous le nom HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Modèles utilisés
CHAT_MODEL = "deepseek-ai/DeepSeek-V3"
VISION_MODEL = "zai-org/GLM-OCR" # Excellent pour l'OCR/Vision

# 2. CONFIGURATION CORS (Pour autoriser ton site Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tu pourras restreindre à ton URL Vercel plus tard
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. ENDPOINT : CHAT & TRADUCTION TEXTE
@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...), 
    mode: str = Form("chat")
):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clé API manquante sur le serveur.")

    # Construction du prompt selon le mode
    system_prompt = "Tu es Llink, l'IA de Gabriel Lab."
    if mode == "translate":
        system_prompt = "Tu es un traducteur expert. Traduis le texte suivant fidèlement. Réponds uniquement avec la traduction."

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Note: On utilise l'URL router pour DeepSeek ou l'URL Inference classique
            url = "https://router.huggingface.co/v1/chat/completions"
            response = await client.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {"response": data["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"response": f"Erreur serveur (Online): {str(e)}"}

# 4. ENDPOINT : OCR & VISION (ANALYSE D'IMAGE)
@app.post("/api/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    img_data = await file.read()
    
    # URL de l'API Inférence
    url = "https://router.huggingface.co/models/zai-org/GLM-OCR"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # On envoie l'image avec les headers d'autorisation
            response = await client.post(url, headers=HEADERS, content=img_data)
            
            # LOG DE SÉCURITÉ : Pour voir ce que Hugging Face répond réellement dans tes logs Render
            print(f"Status Code: {response.status_code}")
            print(f"Raw Response: {response.text}")

            if response.status_code == 503:
                return {"response": "Le modèle OCR est en train de charger sur les serveurs. Réessaie dans 30 secondes ! ⏳"}

            if response.status_code != 200:
                return {"response": f"Erreur Hugging Face : {response.text}"}

            # On vérifie si la réponse est bien du JSON avant de parser
            try:
                result = response.json()
                # Selon le modèle, le texte est souvent dans 'generated_text'
                if isinstance(result, list):
                    text = result[0].get("generated_text", "Aucun texte détecté.")
                else:
                    text = result.get("generated_text", "Analyse terminée.")
                return {"response": text}
            except Exception:
                return {"response": f"Réponse non-JSON reçue : {response.text[:100]}"}

        except Exception as e:
            return {"response": f"Erreur de connexion : {str(e)}"}

@app.get("/")
def home():
    return {"status": "Llink Server is running 🚀"}
