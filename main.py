import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
import firebase_admin
from firebase_admin import credentials, auth

# 1. INITIALISATION DE FIREBASE (SÉCURITÉ)
# Assure-toi d'avoir le fichier firebase-adminsdk.json à la racine de ton projet sur Render
try:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"⚠️ Erreur lors de l'initialisation de Firebase : {e}")

app = FastAPI(title="Llink API Server")

# 2. CONFIGURATION DES VARIABLES D'ENVIRONNEMENT
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

CHAT_MODEL = "deepseek-ai/DeepSeek-V3"
VISION_MODEL = "zai-org/GLM-OCR"

# 3. CONFIGURATION CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # À remplacer par ton URL Vercel en production
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. FONCTION DE VÉRIFICATION DU JETON (NOUVEAU)
async def verify_token(authorization: str = Header(None)):
    """ Vérifie que la requête provient bien d'un utilisateur connecté sur Llink. """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentification requise.")
    
    token = authorization.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token # Contient les infos de l'utilisateur (email, uid, etc.)
    except Exception:
        raise HTTPException(status_code=401, detail="Jeton Firebase invalide ou expiré.")

# 5. ENDPOINT : CHAT & TRADUCTION
@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...), 
    mode: str = Form("chat"),
    user: dict = Depends(verify_token) # EXIGE L'AUTHENTIFICATION ICI
):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clé API Hugging Face manquante.")

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
            url = "https://router.huggingface.co/v1/chat/completions"
            response = await client.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            return {"response": data["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"response": f"Erreur serveur : {str(e)}"}

# 6. ENDPOINT : OCR & VISION
@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    user: dict = Depends(verify_token) # EXIGE L'AUTHENTIFICATION ICI
):
    img_data = await file.read()
    url = f"https://api-inference.huggingface.co/models/{VISION_MODEL}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=HEADERS, content=img_data)
            
            if response.status_code == 503:
                return {"response": "Le modèle OCR charge. Réessaie dans 30 secondes ! ⏳"}
            if response.status_code != 200:
                return {"response": f"Erreur Hugging Face : {response.text}"}

            try:
                result = response.json()
                if isinstance(result, list):
                    text = result[0].get("generated_text", "Aucun texte détecté.")
                else:
                    text = result.get("generated_text", "Analyse terminée.")
                return {"response": text}
            except Exception:
                return {"response": "Réponse non-JSON reçue."}

        except Exception as e:
            return {"response": f"Erreur de connexion : {str(e)}"}

@app.get("/")
def home():
    return {"status": "Llink Server is running securely 🚀"}
