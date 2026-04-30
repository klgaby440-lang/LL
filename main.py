import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx  # Le remplaçant asynchrone de requests

app = FastAPI(title="Llink Backend")
@app.get("/")
async def root():
    return {"status": "Llink API est en ligne !", "message": "Utilisez la route /chat pour communiquer."}
# Configuration CORS pour Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration API
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Formatage spécifique pour Qwen (ChatML)
    prompt = f"<|im_start|>system\nTu es Llink, l'IA créée par Gabriel à Bukavu.<|im_end|>\n<|im_start|>user\n{request.message}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    # On utilise un client asynchrone qui ne bloque pas le serveur
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, headers=headers, json=payload)
            result = response.json()
            
            # Extraction propre de la réponse
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get("generated_text", "")
                reply = full_text.split("assistant\n")[-1].strip()
            else:
                reply = "Désolé, je rencontre une petite erreur technique."
        except Exception as e:
            reply = f"Erreur de connexion : {str(e)}"
            
    return {"reply": reply}
