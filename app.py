from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import ask_question

app = FastAPI()

# Allow your frontend origin
origins = [
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1:5173",
    "https://krishna-portfolio-2.vercel.app/",
    "https://krishnamohan-portfolio2.netlify.app/",
    "https://portfolio-krishna-6rb7dzifs-krishna-mohans-projects-495eaee4.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow your frontend URL
    allow_credentials=True,
    allow_methods=["*"],     # Allow all methods: POST, OPTIONS, GET, etc.
    allow_headers=["*"],     # Allow all headers
)

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    print("User:", message.text)
    reply = ask_question(message.text)
    print("Bot:", reply)
    return {"reply": reply}
