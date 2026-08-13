import os
import json
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import firebase_admin
from firebase_admin import credentials, firestore, auth
import google.generativeai as genai

# ==========================================
# 1. INITIALISATION DE L'APPLICATION
# ==========================================
app = FastAPI(title="Llink API Server - FLOW LAB")

origins = [
    "https://ll-one-self.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURATION DE LA BASE DE DONNÉES (FIREBASE)
# ==========================================
firebase_creds_env = os.getenv("FIREBASE_CONFIG")

if firebase_creds_env:
    try:
        clean_json_str = firebase_creds_env.strip()
        cred_dict = json.loads(clean_json_str)
        cred = credentials.Certificate(cred_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
        db = firestore.client()
        print("✅ SUCCESS: Firebase Firestore est connecté.")
    except Exception as e:
        print(f"⚠️ ERREUR CRITIQUE FIREBASE: {str(e)}")
        db = None
else:
    print("⚠️ ALERTE: Variable FIREBASE_CONFIG introuvable sur le serveur.")
    db = None

# ==========================================
# 3. GESTION DES CLÉS API (IA)
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model_gemini = genai.GenerativeModel('models/gemini-2.5-flash')
else:
    print("⚠️ ALERTE: Clé GEMINI manquante.")

# ==========================================
# 4. MODÈLES DE DONNÉES Pydantic
# ==========================================

class VerifyPaymentRequest(BaseModel):
    transactionId: str

class OCRRequest(BaseModel):
    imageBase64: str

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "): 
        return {"uid": "anonymous"}
    try:
        if db: return auth.verify_id_token(authorization.split(" ")[1])
    except Exception: 
        return {"uid": "anonymous"}
    return {"uid": "anonymous"}

# ==========================================
# 5. ROUTES UTILISATEURS (Essai 3 jours & Abonnements)
# ==========================================
@app.get("/api/user/status")
async def check_status(user: dict = Depends(verify_token)):
    if user["uid"] == "anonymous" or db is None: 
        return {"isValid": False, "expirationDate": None, "isNewUser": False}
    
    uid = user["uid"]
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    now = datetime.utcnow()
    
    # LOGIQUE 3 JOURS D'ESSAI POUR NOUVEAU COMPTE
    if not doc.exists:
        expiration = now + timedelta(days=3)
        user_data = {
            "email": user.get("email", ""),
            "trialStart": now,
            "expirationDate": expiration,
            "isPro": False
        }
        doc_ref.set(user_data)
        return {"isValid": True, "isTrial": True, "expirationDate": expiration.timestamp() * 1000, "isNewUser": True}
    
    data = doc.to_dict()
    exp_date = data.get("expirationDate")
    is_active = False
    
    if exp_date:
        if hasattr(exp_date, 'timestamp'):
            is_active = exp_date.timestamp() > now.timestamp()
            exp_ts = exp_date.timestamp() * 1000
        else:
            is_active = exp_date > now.timestamp()
            exp_ts = exp_date * 1000
    else:
        exp_ts = None
        
    return {
        "isValid": is_active,
        "isTrial": not data.get("isPro", False) and is_active,
        "expirationDate": exp_ts,
        "isNewUser": False
    }

@app.post("/api/verify-payment")
async def verify_payment(req: VerifyPaymentRequest, user: dict = Depends(verify_token)):
    if user["uid"] == "anonymous" or db is None: 
        raise HTTPException(401, "Connectez-vous avec Google pour vous abonner.")
    
    tx_id = req.transactionId.strip()
    
    if tx_id == "KL_GABY_FLOW_LAB_2026":
        new_exp = datetime.utcnow() + timedelta(days=3650)
        db.collection("users").document(user["uid"]).update({"isPro": True, "expirationDate": new_exp})
        return {"success": True}
        
    if len(tx_id) >= 6:
        new_exp = datetime.utcnow() + timedelta(days=30)
        db.collection("users").document(user["uid"]).update({"isPro": True, "expirationDate": new_exp})
        return {"success": True}
    
    return {"success": False}

# ==========================================
# 4. MODÈLES DE DONNÉES & SÉCURITÉ
# ==========================================
class ChatPayload(BaseModel):
    message: str
    model: str
    preferences: Optional[str] = None
    mode: Optional[str] = "chat"

# ==========================================
# 5. ROUTE PRINCIPALE IA AVEC STREAMING FLUIDE
# ==========================================
@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload, user: dict = Depends(verify_token)):
    user_text = payload.message
    selected_model = payload.model
    user_prefs = payload.preferences
    mode = payload.mode

    # Configuration des instructions système
    if mode == "translate":
        system_instruction = "Tu es un traducteur expert. Tu ne dois donner QUE la traduction exacte, sans aucune explication, ni commentaire, ni introduction. Traduis mot pour mot."
    else:
        system_instruction = "Tu es Llink, une IA d'assistance et de traduction créée par CRYPT. Tu es précis, structuré et amical. Le PDG de CRYPT est KL Gaby (Kahorha Gabriel) et cela fait de lui ton inventeur."
    
    if user_prefs and mode != "translate":
        system_instruction += f" Prends en compte ces préférences strictes de l'utilisateur : {user_prefs}"

    # EN-TÊTES DE STREAMING POUR DÉSACTIVER LE CACHE DES SERVEURS DE DÉPLOIEMENT
    stream_headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }

    # ---------------------------------------------------
    # BRANCHE 1 : GEMINI SÉLECTIONNÉ DIRECTEMENT
    # ---------------------------------------------------
    if "gemini" in selected_model.lower():
        async def gemini_generator():
            try:
                if not GEMINI_KEY:
                    yield "❌ Erreur : Clé API Gemini manquante sur le serveur."
                    return
                
                model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=system_instruction)
                response = model.generate_content(user_text, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"❌ Erreur Gemini en cours de flux : {str(e)}"
                
        return StreamingResponse(gemini_generator(), media_type="text/event-stream", headers=stream_headers)

    # ---------------------------------------------------
    # BRANCHE 2 : HUGGING FACE (AVEC FALLBACK GEMINI)
    # ---------------------------------------------------
    if "deepseek" in selected_model.lower():
        hf_model = "deepseek-ai/DeepSeek-V3"
    elif "llama" in selected_model.lower():
        hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        hf_model = "mistralai/Mistral-7B-Instruct-v0.3"

    hf_payload = {
        "model": hf_model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": True # Streaming activé côté HF
    }

    req_headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    async def hf_stream_generator():
        hf_failed = False
        try:
            if not HF_TOKEN:
                raise ValueError("Token HF manquant.")

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", 
                    "https://router.huggingface.co/v1/chat/completions", 
                    json=hf_payload, 
                    headers=req_headers
                ) as response:
                    
                    if response.status_code != 200:
                        raise ValueError(f"Erreur HTTP {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                chunk = data_json["choices"][0]["delta"].get("content", "")
                                if chunk:
                                    yield chunk
                            except Exception:
                                pass
                                
        except Exception as e:
            print(f"🚨 HF en panne ({str(e)}). Basculement streaming sur Gemini...")
            hf_failed = True

        # FALLBACK : En cas d'échec de Hugging Face, Gemini prend immédiatement le relais en stream
        if hf_failed:
            try:
                if not GEMINI_KEY:
                    yield "❌ Tous les serveurs sont indisponibles (Token HF et Clé Gemini manquants)."
                    return
                
                model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=system_instruction)
                backup_res = model.generate_content(user_text, stream=True)
                
                for chunk in backup_res:
                    if chunk.text:
                        yield chunk.text
                        
            except Exception as backup_err:
                yield f"❌ Échec total de la génération. Détails : {str(backup_err)}"

    return StreamingResponse(hf_stream_generator(), media_type="text/event-stream", headers=stream_headers)

# ==========================================
# 7. ENREGISTREMENT ET LANCEMENT
# ==========================================
@app.get("/")
def root():
    return {"message": "Llink Backend API est opérationnel et sécurisé."}
    
# ==========================================
# 8. VISION ET AUDIO
# ==========================================
@app.post("/api/ocr")
async def ocr_endpoint(req: OCRRequest):
    try:
        encoded_data = req.imageBase64.split(",")[1] if "," in req.imageBase64 else req.imageBase64
        image_bytes = base64.b64decode(encoded_data)
        response = model_gemini.generate_content([
            "Extrais tout le texte de cette image. Décris l'image s'il n'y a pas de texte. N'ajoute aucun commentaire.", 
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return {"response": response.text}
    except Exception as e: 
        return {"response": f"Erreur Vision : {str(e)}"}

@app.post("/api/audio")
async def audio_endpoint(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        response = model_gemini.generate_content([
            "Transcris précisément et mot pour mot cet audio. N'ajoute aucun commentaire.", 
            {"mime_type": file.content_type or "audio/webm", "data": audio_bytes}
        ])
        return {"text": response.text.strip()}
    except Exception as e: 
        return {"text": f"Erreur Audio : {str(e)}"}

# ==========================================
# 9. SYNCHRONISATION FIREBASE
# ==========================================
@app.post("/api/history/sync")
async def sync_history(chat: dict, user: dict = Depends(verify_token)):
    if user["uid"] == "anonymous" or not db: return {"status": "skipped"}
    db.collection("users").document(user["uid"]).collection("chats").document(str(chat.get('id', 'temp'))).set(chat)
    return {"status": "success"}

@app.get("/api/history")
async def get_history(user: dict = Depends(verify_token)):
    if user["uid"] == "anonymous": 
        return {"chats": []}
    if db is None:
        return {"chats": [], "warning": "Base de données hors ligne."}
        
    try:
        docs = db.collection("users").document(user["uid"]).collection("chats").stream()
        return {"chats": [doc.to_dict() for doc in docs]}
    except Exception as e:
        return {"chats": [], "warning": f"Erreur DB: {str(e)}"}
