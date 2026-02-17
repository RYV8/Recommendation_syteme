from __future__ import annotations

from typing import Protocol

import pandas as pd


class MovieRepository(Protocol):
    def load_movies(self) -> pd.DataFrame:
        ...


class UserRepository(Protocol):
    def load_users(self) -> pd.DataFrame:
        ...
