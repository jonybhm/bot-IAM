from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="IAM Support Bot", version="0.1.0")
app.include_router(router)

@app.get("/")
async def root():
    return {"status": "ok"}