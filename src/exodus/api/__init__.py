"""FastAPI surface for an exodus node."""

from exodus.api.routes import create_app, exodus_router

__all__ = ["create_app", "exodus_router"]
