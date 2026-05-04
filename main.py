import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
import firebase_admin
from firebase_admin import credentials, auth, firestore
from pydantic import BaseModel
from typing import Optional, List

# 1. INITIALISATION DE FIREBASE & FIRESTORE (SÉCURITÉ)
try:
    firebase_config_raw = os.getenv("FIREBASE_CONFIG")
    if firebase_config_raw:
        firebase_info = json.loads(firebase_config_raw)
        cred = credentials.Certificate(firebase_info)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialisé avec succès via Variable d'Environnement !")
    else:
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialisé via fichier local.")
    
    # On initialise Firestore pour la sauvegarde de l'historique !
    db = firestore.client()
    print("✅ Firestore connecté avec succès !")
        
except Exception as e:
    print(f"⚠️ Erreur lors de l'initialisation de Firebase/Firestore : {e}")

app = FastAPI(title="Luga Link API Server")

# 2. CONFIGURATION DES VARIABLES D'ENVIRONNEMENT
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

CHAT_MODEL = "deepseek-ai/DeepSeek-V3"
VISION_MODEL = "zai-org/GLM-OCR"

# 3. CONFIGURATION CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. MODÈLES DE DONNÉES (PYDANTIC) POUR RÉSOUDRE TES BUGS ---
class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"
    source: Optional[str] = None
    target: Optional[str] = None

class OCRRequest(BaseModel):
    imageBase64: str

class MessageModel(BaseModel):
    id: Optional[int] = None
    chatId: int
    role: str
    text: str

class ChatSessionModel(BaseModel):
    id: int
    title: str
    time: int
    mode: str
    messages: List[MessageModel] = []

# 5. FONCTION DE VÉRIFICATION DU JETON 
async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentification requise.")
    
    token = authorization.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Jeton Firebase invalide ou expiré.")

# 6. ENDPOINT : CHAT & TRADUCTION (DÉSORMAIS EN JSON)
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, user: dict = Depends(verify_token)):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clé API Hugging Face manquante.")

    system_prompt = "Tu es Llink, l'IA de Gabriel Lab. Sois précis, exhaustif et amical."
    if request.mode == "translate":
        system_prompt = f"Tu es un traducteur expert. Traduis le texte fidèlement de la langue '{request.source}' vers '{request.target}'. Réponds uniquement avec la traduction, sans intro."

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        "max_tokens": 2000,
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
            return {"response": f"Erreur serveur HF : {str(e)}"}

# 7. ENDPOINT : OCR & VISION (GÈRE DÉSORMAIS LE BASE64)
@app.post("/api/ocr")
async def ocr_endpoint(request: OCRRequest, user: dict = Depends(verify_token)):
    try:
        # On sépare le préfixe data:image/png;base64, de la vraie donnée
        encoded_data = request.imageBase64.split(",")[1]
        img_data = base64.b64decode(encoded_data)
        
        url = f"https://api-inference.huggingface.co/models/{VISION_MODEL}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=HEADERS, content=img_data)
            
            if response.status_code == 503:
                return {"response": "Le modèle OCR charge. Réessaie dans 30 secondes ! ⏳"}
            if response.status_code != 200:
                return {"response": f"Erreur Hugging Face : {response.text}"}

            result = response.json()
            if isinstance(result, list):
                text = result[0].get("generated_text", "Aucun texte détecté.")
            else:
                text = result.get("generated_text", "Analyse terminée.")
            return {"response": text}

    except Exception as e:
        return {"response": f"Erreur de traitement d'image : {str(e)}"}

# 8. ENDPOINTS DE SYNCHRONISATION D'HISTORIQUE FIRESTORE
@app.post("/api/history/sync")
async def sync_history(chat: ChatSessionModel, user: dict = Depends(verify_token)):
    """ Sauvegarde un chat entier et ses messages sur Firestore """
    try:
        uid = user["uid"]
        # Chemin dans Firebase: users/{uid}/chats/{chatId}
        chat_ref = db.collection("users").document(uid).collection("chats").document(str(chat.id))
        chat_ref.set(chat.model_dump())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(user: dict = Depends(verify_token)):
    """ Récupère tous les chats depuis Firestore """
    try:
        uid = user["uid"]
        chats_ref = db.collection("users").document(uid).collection("chats")
        docs = chats_ref.stream()
        return {"chats": [doc.to_dict() for doc in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Luga Link Server is running securely 🚀"}
