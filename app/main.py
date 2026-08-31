from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import engine
from app.rate_limit import DiffTooLarge, RateLimitExceeded
from app.routers.analyze import router as analyze_router
from app.routers.auth import router as auth_router
from app.routers.review import router as review_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


app = FastAPI(title="Commitor API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(analyze_router, prefix="/analyze", tags=["analyze"])
app.include_router(review_router, prefix="/review", tags=["review"])


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content=exc.body, headers=exc.headers)


@app.exception_handler(DiffTooLarge)
async def diff_too_large_handler(_: Request, exc: DiffTooLarge) -> JSONResponse:
    return JSONResponse(status_code=413, content=exc.body)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
