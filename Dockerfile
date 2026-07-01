FROM python:3.12-slim

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code. NOTE: keep this in sync with the modules main.py imports — insights.py
# powers Evidence Mode and main.py imports it at startup, so it MUST be copied in
# (otherwise the container crashes on boot with ModuleNotFoundError: insights).
COPY main.py insights.py generate_data.py ./

# Generate the smaller 100k-row production dataset at build time, so the image
# is self-contained and no large data file needs to be committed or uploaded.
RUN python generate_data.py --rows 100000 --out data_100k.csv

# Point the API at the production dataset. The CORS origins are intentionally
# left unset here so the image isn't tied to a specific frontend URL — set
# DATAPULSE_CORS_ORIGINS at deploy time to your frontend's URL.
ENV DATAPULSE_DATA_FILE=./data_100k.csv

# Render (and most PaaS) inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
