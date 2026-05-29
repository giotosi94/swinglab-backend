from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.db.mongodb import connect_db, close_db
from app.routes import sectors, assets, scanner, targets, data

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="SwingLab API",
    description="Swing Trading Analysis & POC Scanner",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sectors.router, prefix="/api/sectors", tags=["Sectors"])
app.include_router(assets.router, prefix="/api/assets", tags=["Assets"])
app.include_router(scanner.router, prefix="/api/scanner", tags=["Scanner"])
app.include_router(targets.router, prefix="/api/targets", tags=["Targets"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])

@app.get("/")
def root():
    return {
        "app": "SwingLab",
        "description": "Swing Trading Analysis & POC Scanner",
        "status": "ok",
        "version": "0.2.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}
