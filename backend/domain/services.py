from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd


class MovieProcessor:
    def process(self, data: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class UserProcessor:
    def process(self, data: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class ModelService:
    def predict_users(self, features: pd.DataFrame) -> Sequence:
        raise NotImplementedError

    def predict_movies(self, features: pd.DataFrame) -> Sequence:
        raise NotImplementedError


class RecommendationService:
    def split_users_per_class(self) -> Dict[str, List[pd.Series]]:
        raise NotImplementedError

    def split_movies_per_class(self) -> Dict[str, List[pd.Series]]:
        raise NotImplementedError

    def recommend_for_user(self, user_id: int) -> Dict[str, List[pd.Series]]:
        raise NotImplementedError
