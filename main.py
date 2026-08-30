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
app = FastAPI(title="Llink API Server - CRYPT ENGINE")

origins = [
    "https://ll-one-self.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://klgaby440-lang.github.io",
    "https://class-net-fawn.vercel.app",
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
# 3. CONFIGURATION DES CLÉS API & RESEND HTTP
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Clé API Resend pour l'envoi d'e-mails via HTTPS (Port 443 - Jamais bloqué sur Render)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
TARGET_EMAIL = "klgaby440@gmail.com"

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model_gemini = genai.GenerativeModel('models/gemini-2.5-flash')
else:
    print("⚠️ ALERTE: Clé GEMINI manquante.")

# ==========================================
# 4. SERVICE D'EXPORT EMAIL VIA L'API RESEND (HTTP)
# ==========================================
async def send_email_via_resend(json_content: str, total_count: int):
    """Envoie le fichier JSON du Dataset à klgaby440@gmail.com via l'API HTTP Resend"""
    if not RESEND_API_KEY:
        print("⚠️ [EMAIL] RESEND_API_KEY non configuré sur Render. Impossible d'envoyer l'e-mail.")
        return False
    
    try:
        filename = f"CRYPT_Dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Encodage du fichier JSON en Base64 pour l'attachement de l'API Resend
        encoded_file = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
        
        email_payload = {
            "from": "CRYPT AI Engine <onboarding@resend.dev>", # Adresse par défaut de test Resend
            "to": [TARGET_EMAIL],
            "subject": f"🚀 [CRYPT AI DATASET] Export de {total_count} Entrées - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "html": (
                f"<h3>Salut Gaby ! ⚡</h3>"
                f"<p>Voici le dataset d'entraînement généré automatiquement pour <b>CRYPT AI</b>.</p>"
                f"<ul>"
                f"<li><b>Nombre total d'échanges inclus :</b> {total_count}</li>"
                f"<li><b>Date d'export UTC :</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</li>"
                f"</ul>"
                f"<p>Le fichier JSON complet est joint à cet e-mail. Bon entraînement pour tes modèles ! 🧠🚀</p>"
            ),
            "attachments": [
                {
                    "filename": filename,
                    "content": encoded_file
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.resend.com/emails", json=email_payload, headers=headers)
            
            if response.status_code in [200, 201]:
                print(f"✅ [RESEND EMAIL] Dataset envoyé avec succès à {TARGET_EMAIL} !")
                return True
            else:
                print(f"❌ [RESEND ERREUR] Code {response.status_code} : {response.text}")
                return False

    except Exception as e:
        print(f"❌ [EMAIL ERREUR] Exception critique lors de l'appel HTTP Resend : {str(e)}")
        return False

async def trigger_dataset_export():
    """Consulte toute la BDD Firestore, prépare le dataset JSON et l'envoie par API HTTP"""
    if not db:
        return "❌ Erreur: Base de données Firestore indisponible."
    
    try:
        docs = db.collection("global_datasets").stream()
        dataset = []
        for doc in docs:
            data = doc.to_dict()
            dataset.append({
                "prompt": data.get("prompt"),
                "response": data.get("response"),
                "user_id": data.get("user_id"),
                "source": data.get("source"),
                "preferences": data.get("preferences"),
                "timestamp": str(data.get("timestamp"))
            })
            
        json_str = json.dumps(dataset, ensure_ascii=False, indent=2)
        success = await send_email_via_resend(json_str, len(dataset))
        
        if success:
            return f"✅ Dataset global ({len(dataset)} messages) généré et envoyé avec succès à {TARGET_EMAIL} via Resend !"
        else:
            return f"⚠️ Dataset généré ({len(dataset)} entrées), mais l'envoi HTTP a échoué (Vérifiez la clé RESEND_API_KEY sur Render)."
    except Exception as e:
        return f"❌ Erreur lors de l'export du dataset : {str(e)}"

async def record_exchange_and_check_counter(prompt: str, response: str, user_uid: str, source: str, preferences: str, mode: str):
    """Enregistre l'échange et vérifie si le cap des 50 messages est atteint"""
    if not db:
        return
    
    try:
        # 1. Enregistrement de l'échange dans la collection globale
        exchange_data = {
            "prompt": prompt,
            "response": response,
            "preferences": preferences,
            "user_id": user_uid,
            "source": source,
            "mode": mode,
            "timestamp": datetime.utcnow()
        }
        db.collection("global_datasets").add(exchange_data)
        
        # 2. Mise à jour du compteur global
        counter_ref = db.collection("system_stats").document("dataset_counter")
        counter_doc = counter_ref.get()
        
        if counter_doc.exists:
            new_count = counter_doc.to_dict().get("total_count", 0) + 1
            counter_ref.update({"total_count": new_count})
        else:
            new_count = 1
            counter_ref.set({"total_count": new_count})
            
        print(f"📊 [CRYPT DATASET] Échange enregistré ({source}). Compteur global = {new_count}")
        
        # 3. Déclenchement automatique tous les 50 messages
        if new_count > 0 and new_count % 50 == 0:
            print(f"🚀 Cap des {new_count} messages atteint ! Export automatique par e-mail...")
            await trigger_dataset_export()
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde du dataset : {str(e)}")

# ==========================================
# 5. MODÈLES DE DONNÉES Pydantic
# ==========================================
class VerifyPaymentRequest(BaseModel):
    transactionId: str

class OCRRequest(BaseModel):
    imageBase64: str

class ChatPayload(BaseModel):
    message: str
    model: str
    preferences: Optional[str] = None
    mode: Optional[str] = "chat"
    source: Optional[str] = "soko_master"

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "): 
        return {"uid": "anonymous"}
    try:
        if db: return auth.verify_id_token(authorization.split(" ")[1])
    except Exception: 
        return {"uid": "anonymous"}
    return {"uid": "anonymous"}

# ==========================================
# 6. ROUTES UTILISATEURS
# ==========================================
@app.get("/api/user/status")
async def check_status(user: dict = Depends(verify_token)):
    if user["uid"] == "anonymous" or db is None: 
        return {"isValid": False, "expirationDate": None, "isNewUser": False}
    
    uid = user["uid"]
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    now = datetime.utcnow()
    
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
# 7. ROUTE PRINCIPALE IA AVEC STREAMING ET COLLECTE
# ==========================================
TRIGGER_COMMAND = "llink/_create#_datasets$_for%_cryptÀ_ai_ù"

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload, user: dict = Depends(verify_token)):
    user_text = payload.message
    selected_model = payload.model
    user_prefs = payload.preferences
    mode = payload.mode
    req_source = payload.source or "llink_web"
    uid = user.get("uid", "anonymous")

    user_source_tag = f"anonymous_{req_source}" if uid == "anonymous" else req_source

    stream_headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }

    # Déclenchement manuel via instruction secrète
    if user_text.strip() == TRIGGER_COMMAND:
        async def manual_export_generator():
            yield "⏳ [CRYPT ENGINE] Commande d'exportation détectée. Compilation du dataset et envoi par API Resend...\n\n"
            result_msg = await trigger_dataset_export()
            yield result_msg

        return StreamingResponse(manual_export_generator(), media_type="text/event-stream", headers=stream_headers)

    if mode == "translate":
        system_instruction = "Tu es un traducteur expert. Tu ne dois donner QUE la traduction exacte, sans aucune explication, ni commentaire, ni introduction. Traduis mot pour mot."
    else:
        system_instruction = "Tu es Llink (Luga Link), une IA d'assistance et de traduction créée par CRYPT. Tu es précis et structuré. Ne répond jamais à une question qu'on ne t'a pas demandé. Réponds toujours dans la langue de l'utilisateur. Si on ne te demande pas une longue réponse, donne toujours la réponse la plus courte possible mais la plus efficace. Le PDG de CRYPT (Core Resolution Yield Plateform Technologie) est KL Gaby (Kahorha Gabriel) et cela fait de lui ton inventeur."
    
    if user_prefs and mode != "translate":
        system_instruction += f" Prends en compte ces préférences strictes de l'utilisateur : {user_prefs}"

    if "gemini" in selected_model.lower():
        async def gemini_generator():
            full_reply = ""
            try:
                if not GEMINI_KEY:
                    yield "❌ Erreur : Clé API Gemini manquante sur le serveur."
                    return
                
                model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=system_instruction)
                response = model.generate_content(user_text, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        full_reply += chunk.text
                        yield chunk.text
                        
                await record_exchange_and_check_counter(user_text, full_reply, uid, user_source_tag, preferences=user_prefs, mode=mode)
                
            except Exception as e:
                yield f"❌ Erreur Gemini en cours de flux : {str(e)}"
                
        return StreamingResponse(gemini_generator(), media_type="text/event-stream", headers=stream_headers)

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
        "stream": True
    }

    req_headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    async def hf_stream_generator():
        hf_failed = False
        full_reply = ""
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
                                    full_reply += chunk
                                    yield chunk
                            except Exception:
                                pass
                                
            await record_exchange_and_check_counter(user_text, full_reply, uid, user_source_tag, preferences=user_prefs, mode=mode)

        except Exception as e:
            print(f"🚨 HF en panne ({str(e)}). Basculement streaming sur Gemini...")
            hf_failed = True

        if hf_failed:
            full_reply = ""
            try:
                if not GEMINI_KEY:
                    yield "❌ Tous les serveurs sont indisponibles (Token HF et Clé Gemini manquants)."
                    return
                
                model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=system_instruction)
                backup_res = model.generate_content(user_text, stream=True)
                
                for chunk in backup_res:
                    if chunk.text:
                        full_reply += chunk.text
                        yield chunk.text
                        
                await record_exchange_and_check_counter(user_text, full_reply, uid, user_source_tag, preferences=user_prefs, mode=mode)

            except Exception as backup_err:
                yield f"❌ Échec total de la génération. Détails : {str(backup_err)}"

    return StreamingResponse(hf_stream_generator(), media_type="text/event-stream", headers=stream_headers)

# ==========================================
# 8. AUTRES ROUTES ET SERVICES
# ==========================================
@app.get("/")
def root():
    return {"message": "Llink Backend API est opérationnel et sécurisé."}

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
