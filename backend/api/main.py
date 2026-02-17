from fastapi import FastAPI, HTTPException

from ..core.config import SETTINGS
from ..domain.schemas import RecommendationResponse
from ..infrastructure.models import JoblibModelService
from ..infrastructure.processors import PandasMovieProcessor, PandasUserProcessor
from ..infrastructure.repositories import CsvMovieRepository, CsvUserRepository
from ..services.recommendations import RecommendationService

app = FastAPI()

movie_repository = CsvMovieRepository(SETTINGS.data_dir / "movies_dataset_uncleaned.csv")
user_repository = CsvUserRepository(SETTINGS.data_dir / "user_dataset_uncleaned.csv")
movie_processor = PandasMovieProcessor()
user_processor = PandasUserProcessor()
model_service = JoblibModelService(
    SETTINGS.model_dir / "model_user.jlp",
    SETTINGS.model_dir / "model_movie.jlp",
)
recommendation_service = RecommendationService(
    movie_repository=movie_repository,
    user_repository=user_repository,
    movie_processor=movie_processor,
    user_processor=user_processor,
    model_service=model_service,
)


@app.get("/movies")
def get_movie(offset: int = 0, limit: int = 100):
    if offset < 0 or limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Invalid pagination params")
    movies = movie_repository.load_movies()
    page = movies.iloc[offset : offset + limit]
    return page.to_dict(orient="records")


@app.get("/users")
def get_user(offset: int = 0, limit: int = 100):
    if offset < 0 or limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Invalid pagination params")
    users = user_repository.load_users()
    page = users.iloc[offset : offset + limit]
    return page.to_dict(orient="records")


@app.post("/movies/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, limit: int = 50, users_limit: int = 50):
    try:
        result = recommendation_service.recommend_for_user(
            user_id,
            limit=limit,
            users_limit=users_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result