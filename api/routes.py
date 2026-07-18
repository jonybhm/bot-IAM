from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: acá va a ir la lógica de memoria + RAG + LLM
    return ChatResponse(response=f"Recibido: {request.message}", sources=[])