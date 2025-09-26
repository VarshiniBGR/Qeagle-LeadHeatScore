from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.config import settings
from app.db import connect_to_mongo, close_mongo_connection
from app.api.routes import router
from app.utils.logging import setup_logging, get_logger
from app.utils.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from app.services.classifier import classifier


# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Lead HeatScore API")
    
    # Connect to MongoDB
    await connect_to_mongo()
    
    # Load trained model if available
    classifier.load_model()
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Lead HeatScore API")
    await close_mongo_connection()
    logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="AI-Powered Sales Lead Classification & Next Action Agent",
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "request_id": request_id
        }
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Lead HeatScore API",
        "version": settings.api_version,
        "status": "running",
        "docs": "/docs"
    }


# Health check endpoint (without version prefix)
@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
