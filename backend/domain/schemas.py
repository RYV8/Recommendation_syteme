from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    recommended_movies: List[Any]
    users_class: List[Any]
    user_class_name: Any
