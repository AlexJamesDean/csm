import os
from typing import List, Dict, Optional, Union
import torch
import torchaudio
import base64
import io
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from generator import load_csm_1b, Segment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="CSM Speech API", description="API for the Conversational Speech Model")

# Global generator object
generator = None

# Initialize device
if torch.cuda.is_available():
    device = "cuda"
    logger.info("Using CUDA for inference")
else:
    device = "cpu"
    logger.info("Using CPU for inference (CUDA not available)")

# Model loading happens on first request to avoid slowing startup
def load_model():
    global generator
    if generator is None:
        logger.info("Loading CSM model...")
        generator = load_csm_1b(device=device)
        logger.info("CSM model loaded successfully")
    return generator

# Pydantic models for API requests
class SpeechRequest(BaseModel):
    text: str
    speaker_id: int = 0
    context: Optional[List[Dict[str, Union[str, int, str]]]] = None
    max_audio_length_ms: int = 10000
    temperature: float = 0.9
    topk: int = 50

class ConversationRequest(BaseModel):
    utterances: List[Dict[str, Union[str, int]]]
    initial_context: Optional[List[Dict[str, Union[str, int, str]]]] = None
    max_audio_length_ms: int = 10000
    temperature: float = 0.9
    topk: int = 50

def process_audio_data(audio_data_base64):
    """Process base64 encoded audio data into a tensor"""
    audio_bytes = base64.b64decode(audio_data_base64)
    audio_io = io.BytesIO(audio_bytes)
    audio_tensor, sample_rate = torchaudio.load(audio_io)
    audio_tensor = audio_tensor.squeeze(0)
    return audio_tensor, sample_rate

def prepare_segment(text, speaker_id, audio_data_base64=None):
    """Prepare a segment from text and optional audio data"""
    model = load_model()
    
    if audio_data_base64:
        audio_tensor, sample_rate = process_audio_data(audio_data_base64)
        # Resample if needed
        if sample_rate != model.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=model.sample_rate
            )
        return Segment(text=text, speaker=speaker_id, audio=audio_tensor)
    return None

@app.on_event("startup")
async def startup_event():
    # Load model in background to avoid blocking startup
    load_model()

@app.get("/")
async def root():
    return {"message": "CSM Speech API", "status": "active"}

@app.post("/generate")
async def generate_speech(request: SpeechRequest):
    """Generate speech from text with optional context"""
    try:
        model = load_model()
        
        # Prepare context segments if provided
        context_segments = []
        if request.context:
            for ctx in request.context:
                segment = prepare_segment(ctx["text"], ctx["speaker_id"], ctx.get("audio_base64"))
                if segment:
                    context_segments.append(segment)
        
        # Generate speech
        audio_tensor = model.generate(
            text=request.text,
            speaker=request.speaker_id,
            context=context_segments,
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk
        )
        
        # Convert to WAV and return
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), model.sample_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_conversation")
async def generate_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Generate a complete conversation with multiple utterances"""
    try:
        model = load_model()
        
        # Prepare initial context
        context_segments = []
        if request.initial_context:
            for ctx in request.initial_context:
                segment = prepare_segment(ctx["text"], ctx["speaker_id"], ctx.get("audio_base64"))
                if segment:
                    context_segments.append(segment)
        
        # Generate each utterance
        generated_segments = []
        for utterance in request.utterances:
            audio_tensor = model.generate(
                text=utterance["text"],
                speaker=utterance["speaker_id"],
                context=context_segments + generated_segments,
                max_audio_length_ms=request.max_audio_length_ms,
                temperature=request.temperature,
                topk=request.topk
            )
            generated_segments.append(
                Segment(text=utterance["text"], speaker=utterance["speaker_id"], audio=audio_tensor)
            )
        
        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        
        # Convert to WAV and return
        buffer = io.BytesIO()
        torchaudio.save(buffer, all_audio.unsqueeze(0).cpu(), model.sample_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    
    except Exception as e:
        logger.error(f"Error generating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": generator is not None}