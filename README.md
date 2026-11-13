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

The browser UI is available at `/`, `/preview/<model_id>`, `/serving` and
`/monitoring` (plus `/serving/<endpoint_id>` and `/monitoring/<endpoint_id>`).
`/docs` is the FastAPI Swagger UI and `/api/openapi.json` the generated OpenAPI
document. Assets live in `models/` and are served with `ETag` and `Cache-Control`
headers.

The interface follows one design system inspired by the Databricks platform:

| File | Purpose |
| --- | --- |
| `models/app.css` | Shared design tokens and components (top bar, sidebar, panels, tables, badges, buttons, forms, drawer, dark theme, responsive rules). |
| `models/index.html` | Catalog workspace: collapsible navigation, KPI tiles, table and card views, filters, an info panel for descriptions and a model details drawer. |
| `models/catalog.js` | Catalog behaviour (search, facets, ordering, URL state, drawer). |
| `models/preview.html`, `models/preview.css` | Model page rendered by the server from the introspection catalog. |
| `models/preview.js` | Playground, method invocation and saved-model loading on the model page. |
| `models/serving.html`, `models/serving.js` | Model Serving workspace: endpoint list, deployment wizard, inference playground and generated API docs. |
| `models/monitoring.html`, `models/monitoring.js` | Model Monitoring dashboard: KPIs, latency, errors, usage and endpoint-level analytics. |
| `models/charts.js` | Dependency-free SVG/HTML chart library (lines, stacked bars, horizontal bars, donuts). |
| `models/platform.css` | Serving/monitoring components layered on the shared design tokens. |

The catalog is re-introspected when a model source changes (`LM_CATALOG_TTL`), so
editing `likelihood/models` is reflected without restarting the server.

### Model Serving and Model Monitoring

Two additional workspaces extend the catalog. They reuse the **same** introspection
catalog, the sandboxed child-process invocation (`server_core.run_invoke`) and the
saved-model loaders (`server_core.prepare_load` / `run_load`): the serving layer does
not duplicate model-loading logic.

| Page | Purpose |
| --- | --- |
| `/serving` | Deploy an invokable method as an HTTP endpoint, list endpoints and manage their lifecycle. |
| `/serving/<endpoint_id>` | Endpoint detail: configuration, deployment log, inference playground, activity and generated API usage. |
| `/monitoring` | Global dashboard over every inference served by the endpoints. |
| `/monitoring/<endpoint_id>` | KPIs, latency, errors, usage patterns and request history of one endpoint. |

The deployment wizard walks through four steps: select a model from the catalog,
optionally load a trained artifact (loader + path, with the in-app file browser),
select an inference method (only JSON-callable methods that do not train/build the
model), and configure the endpoint name, timeout and concurrency. The configuration
is validated before creation; the artifact is then loaded in a background thread and
the endpoint reports `deploying` until it becomes `running` (or `failed` with the real
error and log). Endpoints support stop, restart/start and delete.

Each endpoint exposes `POST /api/serving/endpoints/<id>/predict`, accepts the
`{"input": ...}` shorthand for single-argument methods, `{"inputs": {...}}` or
`{"arguments": {...}}`, and returns the JSON projection of the method result plus a
`request_id` and the measured latency. Every call (successful or failed) appends one
line to `events.jsonl`; that file is the **only** source of the monitoring metrics.
Nothing is fabricated: an empty store renders zero KPIs and explicit empty states.

The API usage tab is generated from the deployed method (`GET
/api/serving/endpoints/<id>/docs`): Python, cURL, JavaScript and raw HTTP samples, the
real parameter schema, the payload example, the expected response and the Bearer
token instructions when `LM_API_TOKEN` protects the API. Samples and copy buttons are
dynamic; there is no hardcoded model or endpoint name.

Deployments and events are persisted in `output/serving/` (override with
`LM_SERVING_STORE`). `deployments.json` survives a restart; the event log is a bounded
JSONL file that is rotated when it grows beyond 16 MiB. Inference runs in short-lived
child processes, so per-endpoint CPU/GPU sampling is not available: the resources panel
reports the server process (when the optional `psutil` package is installed) and states
the limitation explicitly instead of inventing numbers. Every prediction starts the
same sandboxed child interpreter as `POST /api/models/<id>/invoke`; for TensorFlow or
Torch models the first call of each request includes the import time, so size the
endpoint `timeout_seconds` (and `LM_INVOKE_TIMEOUT`) accordingly.

### Playground defaults and example inputs

The Playground shows each constructor parameter with the value the model really
uses: the default recorded in the source, the catalogue sample when the parameter
is required and has no default, and an empty field when it is optional and has no
default. The **Datos de ejemplo** button (`POST /api/models/<id>/sample`) returns
exactly the same values for the arguments of the selected method, so an example
never replaces a configured default:

- a declared default is sent untouched, `0`, `False`, `""` and `[]` included;
- a default that JSON cannot carry (`np.array([])`, `nn.SELU()`) is ignored, so the
  field keeps an editable sample instead of an unusable string;
- an optional argument without a default is left out, so the method keeps its own
  default instead of receiving an invented value;
- only a required argument without a default is generated, shaped by the
  constructor values (`input_shape_parm`, `state_dim`, `n_features`, `n_states`,
  `n_observations`…).

Arrays and tensors (`np.ndarray`, `torch.Tensor`, `tf.Tensor`, `List[...]`) count as
JSON-friendly: the invoke tab suggests a matrix shaped by the constructor and the
sandbox builds the declared object (a `torch.Tensor` for torch annotations), so
methods such as `MultiBanditNet.forward(state)` can be called from the page.

The page keeps the two constructor forms (Playground and the invoke tab) in sync
and regenerates the examples while you edit them: the suggested inputs always
match the values used by the next request, and loading a saved model or picking
another method rebuilds them with the new defaults.

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
* Runs started from the UI (`/api/playground`, `/invoke` and the sessions) cap
  the address space of the child process with `LM_PLAYGROUND_MEMORY`
  (8 GiB by default, `0` disables it). Importing TensorFlow maps ~2.5 GiB on its
  own: a lower value fails with `libtensorflow_cc.so.2: failed to map segment
  from shared object`.
* The model page submits every invocation as a background job
  (`LM_INVOKE_JOB_TIMEOUT`, 15 min) and polls it, so a cold TensorFlow import
  no longer hits the 60 s limit of the synchronous route. The compose file also
  sets `LM_INVOKE_JOB_RETRIES=1` (retry a timed out job with exponential
  backoff); the code default is `0`.
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
| `LM_SERVING` | Enable Model Serving and Model Monitoring (`0`/`--no-serving` keeps only the catalog). | true |
| `LM_SERVING_STORE` | Directory that keeps `deployments.json` and `events.jsonl`. | `output/serving` |
| `LM_SERVING_TIMEOUT` | Default per-request timeout of a deployed endpoint, in seconds (capped by `LM_INVOKE_MAX_TIMEOUT`). | `LM_INVOKE_TIMEOUT` |
| `LM_SERVING_CONCURRENCY` | Default simultaneous predictions per endpoint. | 1 |
| `LM_SERVING_DEPLOY_TIMEOUT` | Seconds the background saved-model validation/load may take. | 600 |
| `LM_SERVING_WORKERS` | Background workers that validate/load deployments. | 2 |
| `LM_SERVING_MAX_EVENTS` | Inference events kept in memory and reloaded from the log. | 20000 |
| `LM_SERVING_LOG_INPUTS` | Store a short input/output preview in each monitoring event (`0` disables it). | true |
| `LM_PLAYGROUND_MEMORY` | Address space (MiB) each sandboxed model process may map; raise it for heavy models or set `0` to disable the cap. | 8192 |
| `LM_INVOKE_TIMEOUT` | Seconds a synchronous `/invoke` or session call may take. | 60 |
| `LM_INVOKE_JOB_TIMEOUT` | Seconds a background invocation (`/invoke/jobs`) may take. | 900 |
| `LM_INVOKE_MAX_TIMEOUT` | Upper bound of the `timeout` a client may request in the body. | 3600 |
| `LM_INVOKE_JOB_RETRIES` | Retries of a background invocation that timed out or whose worker died (`0` disables them). | 0 |
| `LM_INVOKE_JOB_BACKOFF` / `LM_INVOKE_JOB_BACKOFF_MAX` | First pause (seconds) between retries and its ceiling; the pause doubles on every attempt. | 2 / 30 |
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

The negated flags (`--no-playground`, `--no-invoke`, `--no-serving`) turn a feature
off; the equivalent environment variables do the same with a value of `0`
(`LM_PLAYGROUND=0`, `LM_INVOKE=0`, `LM_SESSIONS=0`, `LM_SERVING=0`).

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
POST /api/models/{id}/invoke/jobs     # asynchronous call -> 202 + job_id
POST /api/models/{id}/sample
POST /api/playground
GET  /api/playground/jobs
POST /api/playground/jobs
GET  /api/playground/jobs/{job_id}
DELETE /api/playground/jobs/{job_id}
GET  /api/jobs                        # every job (?kind=invoke|playground)
GET  /api/jobs/{job_id}
DELETE /api/jobs/{job_id}
GET  /api/models/{id}/sessions
POST /api/models/{id}/sessions
POST /api/sessions/{session_id}/invoke
DELETE /api/sessions/{session_id}
GET  /api/notebook
```

Model Serving and Model Monitoring add:

```text
GET    /api/serving/overview
GET    /api/serving/endpoints                  # ?q=&status=&model_id=
POST   /api/serving/endpoints                  # deploy -> 202
POST   /api/serving/validate                   # validate without creating
GET    /api/serving/endpoints/{id}
GET    /api/serving/endpoints/{id}/docs        # generated code samples/schema
GET    /api/serving/endpoints/{id}/events      # ?limit=&status=
POST   /api/serving/endpoints/{id}/predict     # inference
POST   /api/serving/endpoints/{id}/stop
POST   /api/serving/endpoints/{id}/restart
POST   /api/serving/endpoints/{id}/start       # alias of restart
DELETE /api/serving/endpoints/{id}

GET    /api/monitoring/overview                # ?range=&start=&end=&model_id=&endpoint_id=
GET    /api/monitoring/endpoints               # per-endpoint aggregates
GET    /api/monitoring/models                  # per-model aggregates
GET    /api/monitoring/events                  # ?range=&status=&limit=&offset=
GET    /api/monitoring/endpoints/{id}          # endpoint-level analytics + history
GET    /api/monitoring/resources               # host sample / explicit limitation
```

`range` accepts `1h`, `24h`, `7d`, `30d` or `custom` (then `start`/`end` are
ISO-8601 or epoch seconds). The monitoring answers aggregate `events.jsonl`; no
metric is generated when there is no traffic.

### Deploying and invoking an endpoint

```bash
# 1. Deploy a method of the catalog (the backend validates the constructor values)
curl -s -X POST localhost:8000/api/serving/endpoints \
  -H 'Content-Type: application/json' \
  -d '{
        "model_id": "HMM",
        "method": "forward",
        "endpoint_name": "hmm-forward",
        "parameters": {"n_states": 3},
        "config": {"timeout_seconds": 60, "max_concurrency": 2},
        "load": {"enabled": false}
      }'

# 2. Poll until status is running
curl -s localhost:8000/api/serving/endpoints/hmm-forward

# 3. Call it with the single-argument shorthand, `inputs` or `arguments`
curl -s -X POST localhost:8000/api/serving/endpoints/hmm-forward/predict \
  -H 'Content-Type: application/json' \
  -d '{"input": [0, 1, 2, 1, 0]}'

# 4. Generated API documentation (Python, cURL, JavaScript, HTTP)
curl -s localhost:8000/api/serving/endpoints/hmm-forward/docs

# 5. Monitoring
curl -s 'localhost:8000/api/monitoring/overview?range=24h'
```

A deployment created with `load.enabled=true` stores the loader and path, calls the
existing saved-model loader during deployment, and uses that loaded instance for
every prediction. `loaded_artifact` in the endpoint payload records the loader, path,
type and load time. Set `LM_API_TOKEN` to require `Authorization: Bearer <token>` on
every serving and monitoring route (the generated docs include the header).

### Long running calls

A model that imports TensorFlow needs minutes on a cold container, which does
not fit in one request. Every call accepts a `timeout` (seconds, capped by
`LM_INVOKE_MAX_TIMEOUT`) and `POST /api/models/{id}/invoke/jobs` runs it in the
background with `LM_INVOKE_JOB_TIMEOUT` (900 s by default):

```bash
curl -s -X POST localhost:8000/api/models/Arima/invoke/jobs \
  -H 'Content-Type: application/json' \
  -d '{"method": "fit", "arguments": {"data": [...]}, "timeout": 900}'
# {"success": true, "job_id": "…", "status": "pending", "poll_url": "/api/jobs/…"}

curl -s localhost:8000/api/jobs/<job_id>    # poll until status is final
```

`LM_INVOKE_JOB_RETRIES` retries a job that timed out (or whose worker died)
with exponential backoff (`LM_INVOKE_JOB_BACKOFF`, capped by
`LM_INVOKE_JOB_BACKOFF_MAX`) and doubles the time budget on every attempt. The
model page uses this endpoint automatically, so its progress survives a page
reload.

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
python3 -m pytest tests/test_serving.py -v        # serving lifecycle, inference, docs and monitoring
python3 -m pytest tests/test_hmm.py -v            # model tests (requires the full extras)
```

`tests/test_serving.py` builds the app over a temporary introspected model package
and exercises deployment, prediction, the generated docs, saved-model loading,
stop/restart/delete and the monitoring aggregates through `httpx.ASGITransport`
(``TestClient`` is not used because the pinned Starlette predates the installed
httpx).

`tests/test_server_fastapi.py` drives the FastAPI application through its
`TestClient` (install the `tests` extra for `pytest` and `httpx`). It covers the
browser pages, static assets, JSON projections, caching (`ETag`s, `If-None-Match`
lists, `If-Modified-Since`), error pages, the OpenAPI document, `HEAD` and gzip,
the security headers, the Prometheus endpoint, the playground subprocess,
asynchronous jobs (including cancellation), method invocation (including
`input_shape_parm`, tuple conversion and session rebuilds), authentication, the
cache policy of token protected answers, rate limiting and CORS.

