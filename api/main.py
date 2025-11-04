from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import time
import uuid
import logging
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI FastAPI MLOps",
    description="Production-ready AI service with SOTA models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class or result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model: str = Field(..., description="Model used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text")
    task: str = Field(default="sentiment", description="Task type: sentiment, classification")
    model: str = Field(default="distilbert-base-uncased", description="Model to use")

    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('Text cannot be empty')
        return v.strip()


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str


# Startup time
startup_time = time.time()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint - API information"""
    return {
        "message": "AI FastAPI MLOps Service",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )


@app.post("/predict/vision", response_model=PredictionResponse)
async def predict_vision(
    file: UploadFile = File(..., description="Image file for classification")
):
    """
    Vision model inference endpoint.

    Note: This is a demo endpoint. In production, load actual models.
    """
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, etc.)"
            )

        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )

        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 10MB limit"
            )

        # Validate image can be opened
        try:
            image = Image.open(BytesIO(contents))
            image.verify()
            logger.info(f"Processing image: {file.filename}, size: {len(contents)} bytes, format: {image.format}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )

        # TODO: Replace with actual model inference
        # Example: prediction = model.predict(image)
        inference_time = (time.time() - start_time) * 1000

        response = PredictionResponse(
            prediction="demo_prediction",
            confidence=0.85,
            model="vit-base-patch16-224",
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )

        logger.info(f"Vision prediction completed: {request_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vision prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post("/predict/nlp", response_model=PredictionResponse)
async def predict_nlp(request: NLPRequest):
    """
    NLP model inference endpoint.

    Note: This is a demo endpoint. In production, load actual models.
    """
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(f"Processing NLP request: task={request.task}, model={request.model}, text_length={len(request.text)}")

        # TODO: Replace with actual model inference
        # Example: prediction = model.predict(request.text)
        inference_time = (time.time() - start_time) * 1000

        response = PredictionResponse(
            prediction="demo_positive",
            confidence=0.92,
            model=request.model,
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )

        logger.info(f"NLP prediction completed: {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error in NLP prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting AI FastAPI MLOps service...")
    logger.info("Note: Using demo endpoints. Load actual models for production.")
    logger.info("Service ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down AI FastAPI MLOps service...")
    logger.info("Shutdown complete!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
