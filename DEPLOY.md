# Deploying DataPulse

This guide deploys the **backend** (FastAPI + DuckDB) to [Render](https://render.com)
as a Docker web service, and the **frontend** (React + Vite) to
[Vercel](https://vercel.com).

The two are wired together by two environment variables:

| Where    | Variable                 | Points at        |
|----------|--------------------------|------------------|
| Backend  | `DATAPULSE_CORS_ORIGINS` | the Vercel URL   |
| Frontend | `VITE_API_BASE`          | the Render URL   |

Because each side needs the other's URL, deploy in this order: **backend first**,
then frontend, then come back and set the backend's `DATAPULSE_CORS_ORIGINS` to
the real Vercel URL.

Assumed layout: backend in a repo containing `datapulse/`, frontend in a repo
containing `datapulse-frontend/` (they can be the same repo or two repos).

---

## Part 1 — Backend on Render (Docker)

The `Dockerfile` is self-contained: it generates the smaller **100,000-row**
production dataset at build time (`python generate_data.py --rows 100000 --out
data_100k.csv`) and points the API at it via `DATAPULSE_DATA_FILE`. You do **not**
need to commit any data file.

1. Push the backend (`datapulse/` with `Dockerfile`, `render.yaml`, `main.py`,
   `generate_data.py`, `requirements.txt`) to GitHub.

> **Fastest path — Blueprint:** a `render.yaml` is included. In Render choose
> **New → Blueprint**, connect the repo, and Render reads `render.yaml` to create
> the Docker service automatically. It will prompt you for `DATAPULSE_CORS_ORIGINS`
> (set it to your Vercel URL — you can use a placeholder now and update it in
> Part 3). Then skip to step 6 to verify. The manual steps below are the
> alternative if you'd rather click through the dashboard.

2. In the Render dashboard: **New** → **Web Service** → connect the repo.
3. Configure the service:
   - **Runtime:** `Docker` (Render auto-detects the `Dockerfile`).
   - **Root Directory:** `datapulse` (only if the Dockerfile is in a subfolder;
     leave blank if it's at the repo root).
   - **Instance Type:** Free is fine — the 100k dataset uses well under 512 MB.
   - Render injects `$PORT`; the container already binds to `0.0.0.0:$PORT`, so
     no start command override is needed.
4. Add environment variables (**Environment** tab):

   | Key                      | Value                                                | Notes |
   |--------------------------|------------------------------------------------------|-------|
   | `DATAPULSE_CORS_ORIGINS` | `https://YOUR-APP.vercel.app`                         | Comma-separate multiple origins. You'll get the real URL after Part 2 — set a placeholder now and update it. |
   | `DATAPULSE_USE_PARQUET`  | `1`                                                  | *(optional)* query from disk instead of RAM for lower memory. |
   | `DATAPULSE_DATA_FILE`    | `./data_100k.csv`                                    | *(optional)* already set in the Dockerfile; override only to change datasets. |

5. **Create Web Service** and wait for the build/deploy to finish.
6. Verify: open `https://YOUR-API.onrender.com/data/summary` — you should see
   `"total_rows":100000`.

> Note: Render free services spin down when idle, so the first request after a
> while takes ~30–60 s to cold-start.

Record the backend URL, e.g. `https://datapulse-api.onrender.com`.

---

## Part 2 — Frontend on Vercel

The frontend reads its API base URL from `VITE_API_BASE` at **build time**
(`src/api.js`), defaulting to `http://localhost:8000` when unset.

1. Push the frontend (`datapulse-frontend/`) to GitHub.
2. In the Vercel dashboard: **Add New** → **Project** → import the repo.
3. Configure the project:
   - **Framework Preset:** `Vite` (auto-detected).
   - **Root Directory:** `datapulse-frontend` (only if it's in a subfolder).
   - **Build Command:** `npm run build` (default).
   - **Output Directory:** `dist` (default).
4. Add an environment variable (**Settings → Environment Variables**, scope to
   **Production**):

   | Key             | Value                                   |
   |-----------------|-----------------------------------------|
   | `VITE_API_BASE` | `https://datapulse-api.onrender.com`    |

   Use your real Render URL from Part 1, with **no trailing slash**.
5. **Deploy.** Vercel builds and serves the static `dist/` output.

Record the frontend URL, e.g. `https://datapulse-frontend.vercel.app`.

---

## Part 3 — Connect them (CORS)

1. Back in **Render → your service → Environment**, set
   `DATAPULSE_CORS_ORIGINS` to the exact Vercel URL from Part 2 (e.g.
   `https://datapulse-frontend.vercel.app`, no trailing slash). Comma-separate
   if you also want preview/custom domains.
2. Save — Render redeploys automatically.
3. Open the Vercel URL in a browser. The dashboard should load summary stats,
   the table, and charts with **no CORS errors** in the dev console.

> If you change `VITE_API_BASE` later, you must **redeploy the frontend** —
> Vite bakes it into the bundle at build time; it is not read at runtime.

---

## Quick reference — environment variables

**Backend (Render):**
- `DATAPULSE_CORS_ORIGINS` — comma-separated allowed frontend origins **(required in prod)**
- `DATAPULSE_DATA_FILE` — path to the dataset (default `./data_100k.csv` in the image)
- `DATAPULSE_PARQUET_FILE` — Parquet path (defaults to the data file with a `.parquet` extension)
- `DATAPULSE_USE_PARQUET` — `1` to query Parquet from disk (lower memory)
- `PORT` — injected by Render; the container binds to it automatically

**Frontend (Vercel):**
- `VITE_API_BASE` — backend base URL, no trailing slash **(required in prod)**

---

## Local sanity check before deploying

Build the production image and run it exactly as Render will:

```bash
cd datapulse
docker build -t datapulse-api .
docker run --rm -p 8000:8000 \
  -e DATAPULSE_CORS_ORIGINS="http://localhost:5173" \
  datapulse-api
curl localhost:8000/data/summary    # -> "total_rows":100000
```

Build the frontend against that backend:

```bash
cd datapulse-frontend
VITE_API_BASE=http://localhost:8000 npm run build
npm run preview                     # serves dist/ at http://localhost:4173
```
