import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_login import UserMixin  # type: ignore
from nomic import embed  # type: ignore
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

db = SQLAlchemy()


class AdminUser(db.Model, UserMixin):  # type: ignore
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class KnowledgeBaseFile(db.Model):  # type: ignore
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filetype = db.Column(db.String(10), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    filehash = db.Column(db.String(64), nullable=False)


class NomicAtlasEmbeddings(Embeddings):
    """
    Embedding model using Nomic Atlas API.
    Implements the LangChain Embeddings interface.
    """
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "nomic-embed-text-v1.5") -> None:
        self.api_key = api_key or os.getenv("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NOMIC_API_KEY environment variable is required for "
                "Nomic Atlas embeddings. Please add it to your .env and run "
                "'nomic login <api-key>' in your terminal."
            )
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Returns a list of embeddings for a list of texts
        result = embed.text(
            texts=texts,
            model=self.model
        )
        return result["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        # Returns a single embedding for a query string
        result = embed.text(
            texts=[text],
            model=self.model
        )
        return result["embeddings"][0]


def load_llm() -> ChatGoogleGenerativeAI:
    """
    Load and return the Google Gemini LLM instance.
    
    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1
    )


def load_embedding_model() -> NomicAtlasEmbeddings:
    """
    Load and return the Nomic Atlas embedding model instance.
    Returns:
        NomicAtlasEmbeddings: Configured embedding model instance
    """
    return NomicAtlasEmbeddings() 