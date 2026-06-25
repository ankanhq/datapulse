# Deploying DataPulse

DataPulse runs as two pieces: the **backend** (FastAPI + DuckDB) on
[Render](https://render.com) as a Docker service, and the **frontend** (React +
Vite) on [Vercel](https://vercel.com). They're wired together by two environment
variables:

| Where    | Variable                 | Points at        |
|----------|--------------------------|------------------|
| Backend  | `DATAPULSE_CORS_ORIGINS` | the Vercel URL   |
| Frontend | `VITE_API_BASE`          | the Render URL   |

Because each side needs the other's URL, deploy the **backend first**, then the
frontend, then come back and set the backend's `DATAPULSE_CORS_ORIGINS` to the
real Vercel URL.

---

## Part 1 — Backend on Render (Docker)

The `Dockerfile` is self-contained: it generates a 100,000-row sample dataset at
build time and points the sample endpoint at it via `DATAPULSE_DATA_FILE`. No
data file needs to be committed — uploads and pastes don't need one at all.

A `render.yaml` blueprint is included, so the fastest path is:

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, connect the repo. Render reads `render.yaml`
   and creates the Docker web service, prompting you for `DATAPULSE_CORS_ORIGINS`
   (use a placeholder now, e.g. `https://placeholder.vercel.app`, and update it
   in Part 3).
3. Wait for the build to finish and the service to go **Live**.

Prefer to click through manually instead? **New → Web Service**, connect the repo
(Render auto-detects the `Dockerfile`), pick the Free instance, and add the env
var below. The container binds to `0.0.0.0:$PORT`, which Render injects — no start
command needed.

| Key                      | Value                          | Notes |
|--------------------------|--------------------------------|-------|
| `DATAPULSE_CORS_ORIGINS` | `https://YOUR-APP.vercel.app`  | The frontend origin. Comma-separate multiple. Update after Part 2. |

Verify: open `https://YOUR-API.onrender.com/` — you should see
`{"message":"DataPulse API is running","active_datasets":0}`.

> Free Render services spin down when idle, so the first request after a while
> takes ~30–60 s to cold-start.

Record the backend URL, e.g. `https://datapulse-api.onrender.com`.

---

## Part 2 — Frontend on Vercel

The frontend reads its API base URL from `VITE_API_BASE` at **build time**.

1. Push the frontend repo to GitHub.
2. In Vercel: **Add New → Project**, import the repo. The **Vite** preset is
   auto-detected (build `npm run build`, output `dist`).
3. Add an environment variable (scope to **Production**):

   | Key             | Value                                |
   |-----------------|--------------------------------------|
   | `VITE_API_BASE` | `https://datapulse-api.onrender.com` |

   Use your real Render URL from Part 1, **no trailing slash**.
4. **Deploy.**

Record the frontend URL, e.g. `https://datapulse-frontend.vercel.app`.

---

## Part 3 — Connect them (CORS)

1. In **Render → your service → Environment**, set `DATAPULSE_CORS_ORIGINS` to
   the exact Vercel URL from Part 2 (no trailing slash). Save — Render redeploys.
2. Open the Vercel URL, upload a CSV, and confirm it loads with no CORS errors.

> Changing `VITE_API_BASE` later requires a **frontend redeploy** — Vite bakes it
> into the bundle at build time; it is not read at runtime.

---

## Environment variables

**Backend (Render):**
- `DATAPULSE_CORS_ORIGINS` — comma-separated allowed frontend origins **(required in prod)**
- `DATAPULSE_DATA_FILE` — source file for the sample endpoint (default `./data_100k.csv` in the image)
- `DATAPULSE_MAX_UPLOAD_MB` — per-upload size limit, MB (default 25)
- `DATAPULSE_MAX_DATASETS` — max in-memory datasets before LRU eviction (default 8)
- `DATAPULSE_DATASET_TTL` — seconds before an idle dataset is evicted (default 1800)
- `DATAPULSE_SAMPLE_ROWS` — row cap for the sample (default 50000)
- `PORT` — injected by Render; the container binds to it automatically

**Frontend (Vercel):**
- `VITE_API_BASE` — backend base URL, no trailing slash **(required in prod)**

---

## Local check before deploying

```bash
# Backend, exactly as Render runs it
cd datapulse
docker build -t datapulse-api .
docker run --rm -p 8000:8000 -e DATAPULSE_CORS_ORIGINS="http://localhost:5173" datapulse-api

# Frontend against that backend
cd ../datapulse-frontend
VITE_API_BASE=http://localhost:8000 npm run build && npm run preview
```
