import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

oxlo_api_key = os.getenv("OXLO_API_KEY")

client = openai.OpenAI(
    base_url="https://api.oxlo.ai/v1",
    api_key=oxlo_api_key
)

app = FastAPI(title="Chatbot Studio Pilates avec DeepSeek")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


SYSTEM_PROMPT = """
Tu es l'assistant virtuel d'un studio Pilates.

Ton rôle est de répondre aux clients de manière simple, claire et professionnelle.

Informations du studio :
- Nom : Pilates Studio
- Horaires : lundi à samedi, de 8h à 20h
- Cours proposés : Pilates au sol, Pilates Reformer, stretching, renforcement doux
- Niveau : les débutants sont acceptés
- Tarif : à partir de 30 DT par séance
- Réservation : par téléphone ou WhatsApp
- Adresse : centre-ville

Règles :
- Réponds toujours en français.
- Sois poli et rassurant.
- Si tu ne connais pas une information, dis au client de contacter le studio.
- Ne donne pas d'informations inventées.
"""


@app.get("/")
def home():
    return {
        "message": "API Chatbot Studio Pilates fonctionne"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not oxlo_api_key:
        raise RuntimeError("OXLO_API_KEY is not configured")

    response = client.chat.completions.create(
        model="deepseek-r1-8b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": request.message
            }
        ],
        max_tokens=512
    )

    bot_response = response.choices[0].message.content

    return ChatResponse(response=bot_response)
