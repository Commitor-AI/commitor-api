from fastapi import APIRouter, Depends, status

from app.dependencies.auth import get_current_user_from_api_key
from app.models.user import User
from app.rate_limit import analyze_limiter, rate_limit_dependency
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import analyze_diff

router = APIRouter()


@router.post(
    "",
    response_model=AnalyzeResponse,
    dependencies=[Depends(rate_limit_dependency(analyze_limiter))],
)
async def analyze(
    payload: AnalyzeRequest,
    _: User = Depends(get_current_user_from_api_key),
) -> AnalyzeResponse:
    return await analyze_diff(payload)
