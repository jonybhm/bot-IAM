from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse
from rag.pipeline import procesar_mensaje

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    respuesta = procesar_mensaje(request.user_id, request.message)
    return ChatResponse(response=respuesta, sources=[])