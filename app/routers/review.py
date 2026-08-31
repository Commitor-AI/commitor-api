from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user_from_api_key
from app.models.user import User
from app.schemas.review import ReviewRequest, ReviewResponse
from app.services.pr_review import review_hunks

router = APIRouter()


@router.post(
    "",
    response_model=ReviewResponse,
)
async def review(
    payload: ReviewRequest,
    _: User = Depends(get_current_user_from_api_key),
) -> ReviewResponse:
    return await review_hunks(payload)
