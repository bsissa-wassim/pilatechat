import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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

- Description :
Pilates Studio est un espace dédié au bien-être, au renforcement musculaire doux, à la posture et à la remise en forme grâce à des cours de Pilates adaptés à tous les niveaux.

- Cours proposés :
  • Pilates au sol
  • Pilates Reformer
  • Stretching
  • Renforcement doux
  • Pilates débutant
  • Pilates posture
  • Pilates relaxation
  • Cours individuels
  • Cours en groupe

- Niveau :
Les débutants sont acceptés. Les cours sont adaptés selon le niveau du client : débutant, intermédiaire ou avancé.

- Coachs :
  • Coach Amira : spécialisée en Pilates débutant et stretching
  • Coach Nour : spécialisée en Pilates Reformer et renforcement doux
  • Coach Lina : spécialisée en posture, mobilité et relaxation

- Espaces disponibles :
  • Salle Pilates au sol
  • Salle Pilates Reformer
  • Espace stretching et relaxation
  • Accueil client
  • Vestiaires
  • Espace attente

- Réservations :
Les clients peuvent réserver une séance selon la disponibilité des cours, du coach et de l’horaire choisi.

- Programmes proposés :
  • Programme débutant
  • Programme remise en forme
  • Programme posture et mobilité
  • Programme renforcement doux
  • Programme relaxation et bien-être

- Suivi client :
Chaque client peut avoir un profil personnalisé avec ses objectifs, son niveau, ses séances réservées et sa progression.

- Objectifs possibles :
  • Améliorer la posture
  • Gagner en souplesse
  • Renforcer les muscles en douceur
  • Réduire le stress
  • Améliorer la mobilité
  • Reprendre une activité physique progressivement

- Communication :
Le studio peut envoyer des notifications concernant les réservations, les rappels de séances, les changements d’horaire et les nouveaux programmes.

- Paiement :
Le paiement peut être effectué par séance, par abonnement ou par pack de séances.

- Ambiance :
Le studio propose une ambiance calme, professionnelle et motivante, adaptée aux personnes qui cherchent le bien-être et la progression.

Règles :
- Réponds toujours en français.
- Sois poli et rassurant.
- Donne des conseils adaptés au niveau et aux objectifs du client.
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
        max_tokens=120
    )

    bot_response = response.choices[0].message.content

    return ChatResponse(response=bot_response)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not oxlo_api_key:
        raise RuntimeError("OXLO_API_KEY is not configured")

    def generate():
        stream = client.chat.completions.create(
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
            max_tokens=512,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
