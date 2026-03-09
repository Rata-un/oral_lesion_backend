import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.web import router as web_router
from app.api.line import router as line_router


def create_app() -> FastAPI:
    app = FastAPI()

    origins = [
    "http://localhost:5173",
    "https://oral-lesion-9459d.web.app",
    "https://oral-lesion-9459d.firebaseapp.com"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    os.makedirs("uploads", exist_ok=True)

    @app.get("/")
    def health_check():
        return {"status": "i'm ok"}

    app.include_router(web_router)
    app.include_router(line_router)
    return app

app = create_app()
