# DataPulse — Backend

High-performance local analytics engine: **FastAPI + DuckDB**. Serves
summary stats, paginated/filterable rows, and chart aggregations over a
10-million-row dataset, all on a laptop.

## Setup

```bash
cd datapulse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Generate data

```bash
python generate_data.py            # 10M rows -> data_10m.csv (~600 MB)
# smaller production dataset (committed/built into the Docker image):
python generate_data.py --rows 100000 --out data_100k.csv   # ~4.4 MB
```

## 2. Run the API

```bash
uvicorn main:app --reload
```

Interactive docs at http://localhost:8000/docs

## Configuration (environment variables)

| Variable                 | Default                                   | Purpose                                            |
|--------------------------|-------------------------------------------|----------------------------------------------------|
| `DATAPULSE_DATA_FILE`    | `./data_10m.csv`                          | Path to the dataset to serve.                      |
| `DATAPULSE_PARQUET_FILE` | data file with `.parquet` extension       | Parquet path used in `--parquet`/Parquet mode.     |
| `DATAPULSE_USE_PARQUET`  | unset                                     | `1` to query Parquet from disk instead of RAM.     |
| `DATAPULSE_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed frontend origins.      |

For production deployment (Render + Vercel), see [DEPLOY.md](DEPLOY.md). A
`Dockerfile` is included that generates the 100k dataset at build time and runs
the API bound to `$PORT`.

## Design note

Unlike the original guide (which queried the CSV directly on every request),
DataPulse loads the CSV into an in-memory DuckDB table **once at startup**.
For a 10M-row dataset this turns multi-hundred-millisecond CSV re-parses into
millisecond queries.

## Parquet mode (`--parquet`)

```bash
python main.py --parquet         # or: DATAPULSE_USE_PARQUET=1 uvicorn main:app
```

On first run this converts `data_10m.csv` -> `data_10m.parquet` once (skipped
if it already exists), then queries the Parquet file **directly from disk** via
a view instead of loading the whole dataset into RAM. Same four endpoints,
byte-for-byte identical results.

Measured on the 10M-row dataset (462 MB CSV -> 142 MB Parquet):

| Mode    | Startup        | Baseline RSS |
|---------|----------------|--------------|
| CSV     | ~0.9 s         | ~860 MB      |
| Parquet | ~0.25 s        | ~200 MB      |

One-time conversion adds ~1 s to the very first `--parquet` start. Note: heavy
queries (deep sorts, big offsets) still pull data into memory to execute, so
peak RSS converges under load — the win is the resident footprint and cold
start. Results are identical because sorts use `id` as a deterministic
tiebreaker, so low-cardinality sort columns (e.g. `value`) paginate the same
way regardless of storage.

## Endpoints

| Method | Path             | Purpose                                    |
|--------|------------------|--------------------------------------------|
| GET    | `/`              | Health check                               |
| GET    | `/data/summary`  | Row count, columns, time range, avg, cats  |
| GET    | `/data/query`    | Paginated, sortable, filterable rows       |
| GET    | `/data/chart`    | `value_over_time` / `category_distribution`|

### Examples

```bash
curl localhost:8000/data/summary

curl "localhost:8000/data/query?page=1&page_size=20&sort_by=value&sort_order=desc&category=Network&min_value=50"

curl "localhost:8000/data/chart?chart_type=value_over_time&interval=month"
curl "localhost:8000/data/chart?chart_type=category_distribution"
```

## Security

`sort_by` and `interval` are validated against fixed allow-lists; every
user-supplied value is passed as a bound `?` parameter, so the dynamic SQL
is not injectable.
