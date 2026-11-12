![likelihood](https://raw.githubusercontent.com/jzsmoreno/likelihood/main/likelihood.png)

![GitHub last commit](https://img.shields.io/github/last-commit/jzsmoreno/likelihood?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/jzsmoreno/likelihood?style=for-the-badge)
![License](https://img.shields.io/github/license/jzsmoreno/likelihood?style=for-the-badge)

<!-- Project description -->
This repository contains the code to build the [likelihood package](./likelihood/) which contains tools for typical tasks in maintain machine learning models in production and the training of custom models, for more information review our [`documentation`](https://jzsmoreno.github.io/likelihood/).

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a _Windows/Linux/Mac_ machine running [Python 3.10+](https://www.python.org/).
* You have installed the latest versions of [`pip`](https://pip.pypa.io/en/stable/installing/) and [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/) or `conda` ([Anaconda](https://www.anaconda.com/distribution/)).

## Installation

This package can be easily installed with pip:
```bash
pip install likelihood
```

## Examples

You can check the [examples](https://github.com/jzsmoreno/likelihood/tree/main/examples) folder.

More examples will be added soon.

## Model catalog web server

The repository includes a FastAPI catalogue (`server_fastapi.py`) backed by the
shared domain layer in `server_core.py`, which introspects the package sources
through `model_catalog.py`. Nothing is hardcoded: the model list, signatures,
parameters, methods, examples and notebooks are all derived from the real
implementation.

Install the optional server dependencies and start it:

```bash
pip install "likelihood[server]"              # fastapi + uvicorn
uvicorn server_fastapi:app --port 8000        # http://127.0.0.1:8000
python3 server_fastapi.py --port 8080 -v      # custom port, log every request
python3 server_fastapi.py --list              # print the catalog and exit
python3 server_fastapi.py --version           # package version and exit
```

The browser UI is available at `/` and `/preview/<model_id>`. `/docs` is the
FastAPI Swagger UI and `/api/openapi.json` the generated OpenAPI document. Assets
live in `models/` and are served with `ETag` and `Cache-Control` headers.

The interface follows one design system inspired by the Databricks platform:

| File | Purpose |
| --- | --- |
| `models/app.css` | Shared design tokens and components (top bar, sidebar, panels, tables, badges, buttons, forms, drawer, dark theme, responsive rules). |
| `models/index.html` | Catalog workspace: collapsible navigation, KPI tiles, table and card views, filters, an info panel for descriptions and a model details drawer. |
| `models/catalog.js` | Catalog behaviour (search, facets, ordering, URL state, drawer). |
| `models/preview.html`, `models/preview.css` | Model page rendered by the server from the introspection catalog. |
| `models/preview.js` | Playground, method invocation and saved-model loading on the model page. |

The catalog is re-introspected when a model source changes (`LM_CATALOG_TTL`), so
editing `likelihood/models` is reflected without restarting the server.

### Saving and loading models

Models that declare a `load`/`load_model` method are shown as *Cargables* in the
UI: their page offers "Cargar un modelo guardado" and
`POST /api/models/<id>/load` restores the instance from a path.

| Model | Save with | Load with | Artifact |
| --- | --- | --- | --- |
| `HMM`, `Arima`, `FourierRegression` | `save_model()` | `load_model()` | pickle |
| `AutoClassifier`, `EnsembleClassifier` | `save()` | `load()` | directory: JSON metadata + Keras components |
| `SimulationEngine` | `save()` | `load()` | pickle |
| `MultiBanditNet` | `save()` | `load()` | Torch checkpoint (config + `state_dict`) |
| `GANRegressor` | `save_model()` | `load_model()` | Keras artifact (`.keras` archive or SavedModel) |
| `ActionSpace` | `save()` | `load()` | pickle |

The Keras artifacts follow the installed TensorFlow: TF > 2.15 writes a single
`.keras` archive while TF <= 2.15 keeps the `save_format="tf"` SavedModel
directory. `AutoClassifier`, `EnsembleClassifier` and `GANRegressor` switch
automatically and their loaders accept either format.

The path field also has an **Examinar…** button that opens an in-app file
browser (`GET /api/files`). It starts at `LM_LOAD_ROOT` (the repository by
default), never navigates above it, and fills the path with the selected file or
folder, so SavedModel/Keras directories can be picked without typing. Set
`LM_FILE_BROWSER=0` to remove it.

#### The saved model is remembered per page

The model page keeps the artifact you selected (path, loader and the constructor
values used to verify it) in `localStorage` under `lm-model-context:<model_id>`.
On the next visit it restores the field, opens the panel, ticks **Usar el modelo
guardado en las peticiones** and rebuilds the playground, the constructor form
and the example arguments from those values, so the third tab is immediately
usable.

While that checkbox stays ticked, every invocation reuses the artifact: the
session is opened with `{"loader": ..., "path": ...}` instead of constructor
arguments (`POST /api/models/<id>/sessions`), and direct invocations send
`path`/`loader` alongside the method (`POST /api/models/<id>/invoke`). Loaders
that do not need a live instance (`requires_instance` is false) skip the
constructor form; the rest keep sending its values.

**Restablecer modelo** asks for confirmation and then forgets the stored
artifact, closes the open session, clears the results and reloads the catalogue
defaults in the playground and in the two forms. Unticking the checkbox, or
editing the path, keeps the request in constructor mode (and closes the instance
that was built from the old artifact).

`EarlyStopping` is a training callback without a checkpoint of its own, and the
remaining classes are not loadable by design: `Env`, `AutoQL` and `GetInsights`
wrap live objects supplied by the caller, and `OptionCriticEnv` is a simulator
derived from an episode dataset. Their docstrings state the reason.

### Docker

The `Dockerfile` bundles the server, the scientific stack and the catalog
sources; `docker-compose.yml` runs it with the usual `LM_*` settings.

```bash
docker build -t likelihood-catalog .
docker run --rm -p 8000:8000 likelihood-catalog     # http://localhost:8000

# Optional: models that need TensorFlow / PyTorch / scikit-learn
docker build --build-arg INSTALL_FULL_EXTRAS=true -t likelihood-catalog:full .

# Compose (reads a local .env file, if present)
docker compose up --build
LM_API_TOKEN=change-me docker compose up --build     # protected /api
LM_PORT=9000 docker compose up                       # container + published port
```

Notes:

* The image runs as the unprivileged user `likelihood` (uid 1001), so mounted
  model files must be readable by that user.
* Compose mounts `likelihood/models` read-only, which keeps the live catalog
  reload while you edit a model source. Remove the mount to use the copy baked
  into the image.
* `LM_API_TOKEN` stays empty by default (open catalogue). Set it before exposing
  the container beyond localhost.
* The image ships a `HEALTHCHECK` against `/api/health`; `docker ps` shows
  `healthy` once the catalogue is ready.
* Model version history (`GET /api/models/<id>/versions`) is read from `git log`,
  and the build context excludes `.git`, so that endpoint answers `503` inside
  the container. Run the server from a checkout when you need it.

### Environment variables

The server reads optional `LM_*` environment variables (each overridable with
the equivalent `--` CLI flag). They let you secure the API or tune its behaviour:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LM_API_TOKEN` | Protects the `/api` endpoints; clients must send `Authorization: Bearer <token>`. | none |
| `LM_CORS_ORIGINS` | Comma-separated list of origins allowed for CORS. | none |
| `LM_RATE_LIMIT` | Maximum requests per minute per client (`0` disables it). | 0 |
| `LM_TRUST_PROXY` | Trust `X-Forwarded-*` headers when identifying the client address. | false |
| `LM_PLAYGROUND` | Enable the interactive model playground. | true |
| `LM_INVOKE` | Enable invoking model methods (still requires authentication when a token is set). | true |
| `LM_SESSIONS` | Enable long-lived model sessions. | true |
| `LM_JOBS_WORKERS` | Number of workers for asynchronous playground jobs. | 2 |
| `LM_CATALOG_TTL` | Cache time to live, in seconds, before the catalog is re-introspected. | 2 |
| `LM_LOAD_ROOT` | Directory the "Examinar…" browser starts at and can never leave. | repository root |
| `LM_LOAD_RESTRICT` | Reject `POST /load` paths outside `LM_LOAD_ROOT`. | on when `LM_LOAD_ROOT` is set |
| `LM_FILE_BROWSER` | Serve the `GET /api/files` browser used by the "Examinar…" dialog. | true |
| `LM_FILE_BROWSER_LIMIT` | Maximum number of entries returned by one directory listing. | 1000 |
| `LM_GZIP_MIN_BYTES` | Minimum response size (bytes) before gzip compression. | 1024 |
| `LM_HOST` / `LM_PORT` | Bind address and port (equivalently `SERVER_HOST` / `SERVER_PORT`). | `127.0.0.1` / `8000` |

Set `LM_API_TOKEN` to expose the catalogue over a network, then hand it to clients
as a bearer token. Every other variable is optional and follows the same `LM_*`
naming convention.

### Overriding environment variables

Every `LM_*` environment variable can be overridden on the command line with
the equivalent flag, which takes precedence over the environment. The same
flags can also be passed directly to `uvicorn`:

```bash
# Prefer the API token over any LM_API_TOKEN set in the shell
python3 server_fastapi.py --token <my-secret-token>

# Allow the frontend on *.example.com (overrides LM_CORS_ORIGINS)
python3 server_fastapi.py --cors-origin http://*.example.com

# Rate limit every client to 60 requests/minute (overrides LM_RATE_LIMIT)
python3 server_fastapi.py --rate-limit 60

# Trust the reverse proxy (overrides LM_TRUST_PROXY)
python3 server_fastapi.py --trust-proxy

# Disable the playground and invoke (overrides LM_PLAYGROUND=0 / LM_INVOKE=0)
python3 server_fastapi.py --no-playground --no-invoke

# Disable gzip compression (overrides LM_GZIP_MIN_BYTES=0)
python3 server_fastapi.py --no-compression
```

The negated flags (`--no-playground`, `--no-invoke`) turn a feature off; the
equivalent environment variables do the same with a value of `0`
(`LM_PLAYGROUND=0`, `LM_INVOKE=0`, `LM_SESSIONS=0`).

### Endpoints

Beyond the browser pages, the server exposes these JSON endpoints:

```text
GET  /api/categories
GET  /api/models
POST /api/search                  # legacy search
GET  /api/models/{id}/source
GET  /api/models/{id}/versions
POST /api/models/{id}/load
POST /api/models/{id}/invoke
POST /api/models/{id}/sample
POST /api/playground
GET  /api/playground/jobs
POST /api/playground/jobs
GET  /api/playground/jobs/{job_id}
DELETE /api/playground/jobs/{job_id}
GET  /api/models/{id}/sessions
POST /api/models/{id}/sessions
POST /api/sessions/{session_id}/invoke
DELETE /api/sessions/{session_id}
GET  /api/notebook
```

When `LM_API_TOKEN` is set, every `/api` route requires a valid bearer token.

Notebooks are generated from the same introspection catalog:

```bash
python3 notebook_generator.py --all
python3 notebook_generator.py --single Arima
python3 notebook_generator.py --category deep
python3 notebook_generator.py --list
```

Each notebook includes the documentation, the parameter table, an executable
example and `ipywidgets` controls derived from the constructor signature.

## Running the tests

```bash
python3 -m pytest tests/test_server_core.py -v    # catalog, coercion and invoke helpers
python3 -m pytest tests/test_server_fastapi.py -v # FastAPI server, API and playground
python3 -m pytest tests/test_hmm.py -v            # model tests (requires the full extras)
```

`tests/test_server_fastapi.py` drives the FastAPI application through its
`TestClient` (install the `tests` extra for `pytest` and `httpx`). It covers the
browser pages, static assets, JSON projections, caching (`ETag`s, `If-None-Match`
lists, `If-Modified-Since`), error pages, the OpenAPI document, `HEAD` and gzip,
the security headers, the Prometheus endpoint, the playground subprocess,
asynchronous jobs (including cancellation), method invocation (including
`input_shape_parm`, tuple conversion and session rebuilds), authentication, the
cache policy of token protected answers, rate limiting and CORS.

