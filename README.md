# Movie Recommendation System

A machine learning-based movie recommendation system that uses collaborative filtering through user and movie clustering. The system provides a REST API built with FastAPI for easy integration.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Machine Learning Model](#machine-learning-model)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Project Structure](#project-structure)

## 🎯 Overview

This recommendation system uses a dual clustering approach:
- **User Clustering**: Groups users with similar preferences
- **Movie Clustering**: Groups movies with similar characteristics

By matching a user's cluster with movies from similar users' preferred clusters, the system generates personalized recommendations.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │   Movies   │  │    Users     │  │   Recommendations    ││
│  │  Endpoint  │  │   Endpoint   │  │      Endpoint        ││
│  └────────────┘  └──────────────┘  └──────────────────────┘│
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
┌───────▼────────┐              ┌─────────▼────────┐
│ User Clustering│              │ Movie Clustering │
│     Model      │              │      Model       │
│  (model_user)  │              │  (model_movie)   │
└────────────────┘              └──────────────────┘
```

## 🤖 Machine Learning Model

### Model Components

The system uses two independent machine learning models:

1. **User Clustering Model** (`model_user.jlp`)
   - Clusters users based on their rating patterns, preferences, and genre interactions
   - Features:
     - Genre preferences (19 genres: Action, Adventure, Animation, etc.)
     - Tag categories (Genre & Style, Themes & Tropes, Actors & Characters, Viewing & Production)
     - Normalized rating behavior
   
2. **Movie Clustering Model** (`model_movie.jlp`)
   - Clusters movies based on their attributes and characteristics
   - Features:
     - Genre classification (19 genres)
     - Tag categories
     - Rating statistics
     - Relevance scores

### Feature Engineering

#### 1. Genre Processing
Converts multi-genre labels into binary features:
```python
Genres: "Action|Adventure|Sci-Fi"
→ [Action: 1, Adventure: 1, Animation: 0, ..., Sci-Fi: 1, ...]
```

Supported genres:
- Action, Adventure, Animation, Children, Comedy
- Crime, Documentary, Drama, Fantasy, Film-Noir
- Horror, IMAX, Musical, Mystery, Romance
- Sci-Fi, Thriller, War, Western

#### 2. Tag Categorization
User-generated tags are categorized into 4 main groups:
- **Genre & Style**: Action-related, horror, comedy, etc.
- **Themes & Tropes**: Time travel, psychological, dystopia, etc.
- **Actors & Characters**: Director names, character types, etc.
- **Viewing & Production**: Watch context, production quality, etc.

#### 3. Rating Normalization
Ratings are standardized using `StandardScaler`:
```python
normalized_rating = (rating - mean) / std_dev
```
Handles edge cases:
- Missing values → filled with mean
- Zero variance → returns zeros
- Empty data → graceful handling

### Recommendation Algorithm

```
1. Load user profile → Extract features → Predict user cluster
2. Find all users in same cluster
3. Get movies watched by cluster members
4. For each movie:
   - Extract movie features
   - Predict movie cluster
5. Return movies from predicted clusters
6. Deduplicate by movieId
7. Apply pagination/limits
```

### Performance Optimizations

- **Caching**: User/movie data and cluster assignments cached in memory
- **Pagination**: Default limit of 50 movies to prevent large payloads
- **Deduplication**: Ensures unique movieId in recommendations
- **Lazy Loading**: Models loaded once on first request

## 🚀 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Get Movies (Paginated)
```http
GET /movies?offset=0&limit=100
```

**Query Parameters:**
- `offset` (int, default: 0): Starting position
- `limit` (int, default: 100, max: 1000): Number of results

**Response:**
```json
[
  {
    "movieId": 1,
    "title": "Toy Story (1995)",
    "genres": "Adventure|Animation|Children|Comedy|Fantasy",
    "rating": 4.5,
    ...
  },
  ...
]
```

**Example:**
```bash
curl "http://localhost:8000/movies?offset=0&limit=10"
```

---

#### 2. Get Users (Paginated)
```http
GET /users?offset=0&limit=100
```

**Query Parameters:**
- `offset` (int, default: 0): Starting position
- `limit` (int, default: 100, max: 1000): Number of results

**Response:**
```json
[
  {
    "userId": 1,
    "movieId": 123,
    "rating": 4.0,
    "genres": "Action|Thriller",
    ...
  },
  ...
]
```

**Example:**
```bash
curl "http://localhost:8000/users?offset=0&limit=10"
```

---

#### 3. Get Recommendations for User
```http
POST /movies/{user_id}?limit=50&users_limit=50
```

**Path Parameters:**
- `user_id` (int, required): The user ID to get recommendations for

**Query Parameters:**
- `limit` (int, default: 50): Number of recommended movies
- `users_limit` (int, default: 50): Number of similar users to consider

**Response:**
```json
{
  "recommended_movies": [
    {
      "movieId": 456,
      "title": "The Matrix (1999)",
      "genres": "Action|Sci-Fi|Thriller",
      ...
    },
    ...
  ],
  "users_class": [
    {
      "userId": 23,
      "rating": 4.5,
      ...
    },
    ...
  ],
  "user_class_name": "2"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/movies/1?limit=20&users_limit=30"
```

**Error Responses:**

404 Not Found:
```json
{
  "detail": "Utilisateur non trouvé. Ce code sera optimsé pour générer une recommandation même pour un utilisateur non présent dans la base de données"
}
```

400 Bad Request (invalid pagination):
```json
{
  "detail": "Invalid pagination params"
}
```

---

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/RYV8/Recommendation_syteme.git
cd Recommendation_syteme
```

2. **Create virtual environment:**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

4. **Prepare data:**
Place your datasets in `backend/data/`:
- `movies_dataset_uncleaned.csv`
- `user_dataset_uncleaned.csv`

5. **Prepare models:**
Place trained models in `backend/models/`:
- `model_user.jlp`
- `model_movie.jlp`

### Running the Server

```bash
cd backend/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 💻 Usage

### Python Example

```python
import requests

# Get movies
response = requests.get("http://localhost:8000/movies?limit=10")
movies = response.json()

# Get recommendations for user
response = requests.post("http://localhost:8000/movies/1?limit=20")
recommendations = response.json()

print(f"User cluster: {recommendations['user_class_name']}")
print(f"Recommended {len(recommendations['recommended_movies'])} movies")
for movie in recommendations['recommended_movies'][:5]:
    print(f"  - {movie['title']}")
```

### JavaScript/Fetch Example

```javascript
// Get recommendations
fetch('http://localhost:8000/movies/1?limit=20', {
    method: 'POST'
})
.then(response => response.json())
.then(data => {
    console.log('Recommendations:', data.recommended_movies);
    console.log('Similar users:', data.users_class);
});
```

### cURL Examples

```bash
# Get 10 movies
curl "http://localhost:8000/movies?limit=10"

# Get recommendations for user 42
curl -X POST "http://localhost:8000/movies/42?limit=20"

# Get users with pagination
curl "http://localhost:8000/users?offset=100&limit=50"
```

## 📊 Data Requirements

### Movies Dataset Format
```csv
movieId,title,genres,rating,tag,tagId,relevance,tagger_userId,rater_userId
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy,4.5,pixar,1,0.8,123,456
```

**Required columns:**
- `movieId`: Unique movie identifier
- `title`: Movie title with year
- `genres`: Pipe-separated genres
- `rating`: Average rating (optional, will be normalized)
- `tag`: User-generated tag (optional)

### Users Dataset Format
```csv
userId,movieId,rating,genres,user_tag
1,31,2.5,Crime|Drama,smart
```

**Required columns:**
- `userId`: Unique user identifier
- `movieId`: Movie the user interacted with
- `rating`: User's rating
- `genres`: Movie genres
- `user_tag`: User's tag (optional)

## 📁 Project Structure

```
recommendation_systems/
├── README.md
├── LICENSE
├── .gitignore
├── backend/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings and configuration
│   │   └── errors.py              # Custom exceptions
│   ├── data/
│   │   ├── movies_dataset_uncleaned.csv
│   │   └── user_dataset_uncleaned.csv
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── repositories.py        # Data access interfaces
│   │   ├── schemas.py             # Pydantic models
│   │   └── services.py            # Business logic interfaces
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── data_processing.py     # Feature engineering
│   │   ├── models.py              # ML model service
│   │   ├── processors.py          # Data processors
│   │   └── repositories.py        # Data access implementations
│   ├── models/
│   │   ├── model_user.jlp         # User clustering model
│   │   └── model_movie.jlp        # Movie clustering model
│   └── services/
│       ├── __init__.py
│       └── recommendations.py     # Recommendation logic
└── frontend/                       # (Future UI implementation)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Data Paths
DATA_DIR=backend/data
MODEL_DIR=backend/models

# Cache Settings
ENABLE_CACHE=True

# Pagination Defaults
DEFAULT_LIMIT=100
MAX_LIMIT=1000
```

### Model Configuration

Models are loaded automatically from `backend/models/`:
- `model_user.jlp`: Joblib-serialized scikit-learn model for user clustering
- `model_movie.jlp`: Joblib-serialized scikit-learn model for movie clustering

## 🐛 Troubleshooting

### Issue: API responds slowly on first request
**Solution**: Models and data are loaded on first request. Subsequent requests use cache and are faster.

### Issue: sklearn RuntimeWarning about division
**Solution**: Already fixed! The `handle_rating()` function now handles zero-variance data gracefully.

### Issue: Large payload causing timeout
**Solution**: Use pagination parameters:
```bash
curl "http://localhost:8000/movies?limit=50"
```

### Issue: Duplicate movies in recommendations
**Solution**: Already fixed! Movies are deduplicated by `movieId` before returning.

## 🚦 Performance Tips

1. **Use pagination**: Always specify reasonable `limit` values
2. **Cache warmup**: Make a test request on startup to load models
3. **Concurrent requests**: FastAPI handles multiple requests efficiently
4. **Data size**: Keep CSV files optimized (large files now ignored in git)

## 📈 Future Improvements

- [ ] Add user authentication
- [ ] Implement collaborative filtering with matrix factorization
- [ ] Add real-time model updates
- [ ] Create frontend dashboard
- [ ] Add A/B testing framework
- [ ] Implement recommendation explanations
- [ ] Add more sophisticated ranking algorithms
- [ ] Support for new user cold-start problem

## 📝 License

This project is licensed under the terms included in the LICENSE file.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub: https://github.com/RYV8/Recommendation_syteme

---

**Built with:**
- FastAPI for the REST API
- scikit-learn for machine learning models
- pandas for data processing
- joblib for model serialization
- pydantic for data validation
