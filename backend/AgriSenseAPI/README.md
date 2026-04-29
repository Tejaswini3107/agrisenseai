# Python backend (Flask + PostgreSQL)

This folder contains a minimal Flask backend configured to use PostgreSQL via `SQLALCHEMY_DATABASE_URI` (read from `DATABASE_URL`).

Quick start

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `DATABASE_URL` and `OPENWEATHER_API_KEY`.

3. Initialize the database (example using psql):

```bash
# create DB and user as needed, then:
export DATABASE_URL=postgresql://user:password@localhost:5432/agrisense
python -c "from app import create_app; from extensions import db; app=create_app();
with app.app_context(): db.create_all(); print('tables created')"
```

4. Run the server:

```bash
export FLASK_APP=app.py
python app.py
```

Endpoints

- `GET /weather` — list stored weather entries
- `POST /weather` — add a weather entry `{ "city": "City", "temperature": 12.3 }`
- `GET /weather/external?city=...` — fetch current weather from OpenWeatherMap (requires `OPENWEATHER_API_KEY`)
