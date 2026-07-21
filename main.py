from fastapi import FastAPI
from api.routes import router
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="IAM Support Bot", version="0.1.0")
app.include_router(router)
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return {"status": "ok"}