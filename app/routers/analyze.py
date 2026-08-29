from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user_from_api_key
from app.models.user import User
from app.rate_limit import analyze_max_diff_chars, enforce_analyze_rate_limit
from app.rate_limit import DiffTooLarge
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import analyze_diff

router = APIRouter()


@router.post(
    "",
    response_model=AnalyzeResponse,
    dependencies=[Depends(enforce_analyze_rate_limit)],
)
async def analyze(
    payload: AnalyzeRequest,
    user: User = Depends(get_current_user_from_api_key),
) -> AnalyzeResponse:
    # Plan-gated size cap. The schema only rejects above the absolute
    # ceiling; the real Free/Pro limit is enforced here so the message
    # can tell the user to upgrade.
    limit = analyze_max_diff_chars(user.plan)
    if len(payload.diff) > limit:
        raise DiffTooLarge(limit, user.plan)
    return await analyze_diff(payload)
