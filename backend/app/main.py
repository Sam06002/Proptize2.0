from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

from optimizer import PetPoojaOptimizer

app = FastAPI(title="PetPooja Prompt Optimizer")
optimizer = PetPoojaOptimizer()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    query: str

class OptimizeResponse(BaseModel):
    original: str
    optimized: str
    intent: str
    entities: Optional[Dict[str, str]] = None

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Optimize a natural language query for the PetPooja Agent.
    
    Args:
        request: Contains the user's natural language query
        
    Returns:
        Dictionary containing original query, optimized prompt, detected intent, and extracted entities
    """
    try:
        if not request.query or not isinstance(request.query, str):
            raise ValueError("Query must be a non-empty string")
            
        return optimizer.optimize_prompt(request.query)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Failed to process query",
                "message": str(e)
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "petpooja-prompt-optimizer"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
