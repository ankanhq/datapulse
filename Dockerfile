FROM python:3.12-slim

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code. NOTE: keep this in sync with the modules main.py imports — it imports
# insights (Evidence Mode), auth (Supabase token verification) and db (persistent
# metadata store) at startup, so ALL must be copied in or the container crashes on
# boot with ModuleNotFoundError.
COPY main.py insights.py auth.py db.py generate_data.py ./

# Build the "story" sample at image build time so the image is self-contained
# and "Try with sample data" shows a clear, high-confidence Evidence Mode story
# on first click (a rising trend, one obvious outlier, a dominant category).
RUN python generate_data.py --profile story --out data_sample.csv

# Point the API at the sample dataset. The CORS origins are intentionally
# left unset here so the image isn't tied to a specific frontend URL — set
# DATAPULSE_CORS_ORIGINS at deploy time to your frontend's URL.
ENV DATAPULSE_DATA_FILE=./data_sample.csv

# Render (and most PaaS) inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
