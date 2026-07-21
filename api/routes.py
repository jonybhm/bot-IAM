from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from api.schemas import ChatRequest, ChatResponse
from rag.pipeline import procesar_mensaje

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    respuesta = await run_in_threadpool(procesar_mensaje, request.user_id, request.message)
    return ChatResponse(response=respuesta, sources=[])