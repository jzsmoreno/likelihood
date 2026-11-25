# Likelihood Backend — Technical Concepts & Mechanisms

> Exhaustive notes on every technical concept and mechanism in `server/`.
> The goal is to explain **how** and **why**, not just **what**.

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [AST-Based Model Catalog (`model_catalog.py`)](#2-ast-based-model-catalog-model_catalogpy)
3. [Core Domain Layer (`server_core.py`)](#3-core-domain-layer-server_corepy)
   - 3.1 Catalog Store & Refresh
   - 3.2 Metrics & Prometheus Counters
   - 3.3 Parameter Type Classification
   - 3.4 Loader Resolution
   - 3.5 Sandboxed Subprocess Execution
   - 3.6 Bootstrap Strings
   - 3.7 `run_invoke()` & `InvocationError`
   - 3.8 `ModelSession` — Persistent Workers
   - 3.9 `JobQueue` — Background Execution
   - 3.10 HTML Rendering Helpers
4. [FastAPI Server (`server_fastapi.py`)](#4-fastapi-server-server_fastapipy)
   - 4.1 `ServerSettings` & Environment Flags
   - 4.2 `ServerState` & Lifecycle
   - 4.3 `_AliasedBody` & Legacy Compatibility
   - 4.4 Endpoint Routing
   - 4.5 CORS, Gzip, Proxy Trust
   - 4.6 Prometheus Middleware & Rate Limiting
   - 4.7 Error Handling & Authentication
5. [Serving Core (`serving_core.py`)](#5-serving-core-serving_corepy)
   - 5.1 `ServingStore` & Concurrency Model
   - 5.2 `EventLog` — Append-Only JSON Lines
   - 5.3 Event Schema Migration
   - 5.4 Model Record Normalisation & Grouping
   - 5.5 Deployment Lifecycle
   - 5.6 Artifact Tracking & Sample Hints
   - 5.7 Error Recording
   - 5.8 API Documentation Generation
6. [Serving API (`serving_api.py`)](#6-serving-api-serving_apipy)
   - 6.1 Deployment Endpoints
   - 6.2 Predict Endpoint
   - 6.3 Monitoring & Events Endpoints
   - 6.4 `deployment_payload()` & Per-Endpoint Stats
7. [Training Core (`training_core.py`)](#7-training-core-training_corepy)
   - 7.1 Training Architecture & Job Types
   - 7.2 `TrainingStore` & Persistence
   - 7.3 Framework Fit Detection (Keras Adapter)
   - 7.4 Job Submission & Normalisation
   - 7.5 Cross-Validation Integration
   - 7.6 Artifact Management & Atomic Swap
   - 7.7 `_run_fit` — Full Training Pipeline
   - 7.8 `_run_evaluate` — Scoring Pipeline
   - 7.9 Series Analysis & Overwrite
   - 7.10 Error Handling in Job Execution
   - 7.11 `_atomic_write` & `_copy_tree`
8. [Training API (`training_api.py`)](#8-training-api-training_apipy)
   - 8.1 Router Design & Reuse of `server_core` Errors
   - 8.2 Job Payload & Summary
   - 8.3 Series Analysis Request Model
   - 8.4 Dataset Upload & Management
9. [Training Datasets (`training_datasets.py`)](#9-training-datasets-training_datapy)
   - 9.1 Format Detection & Limits
   - 9.2 `DatasetStore` & In-Memory Storage
   - 9.3 Column Metadata & Preview
   - 9.4 Sampling & Time-Series Preservation
   - 9.5 Remote URL Fetching & IP Validation
10. [Training Metrics (`training_metrics.py`)](#10-training-metrics-training_metricspy)
11. [Cross-Validation (`cross_validation.py`)](#11-cross-validation-cross_validationpy)
12. [Monitoring Core (`monitoring_core.py`)](#12-monitoring-core-monitoring_corepy)
    - 12.1 Time Range Presets & Buckets
    - 12.2 `aggregate()` — The Metrics Engine
    - 12.3 `range_window()` & `bucketize()`
    - 12.4 `deployment_overview()`
13. [Notebook Generator (`notebook_generator.py`)](#13-notebook-generator-notebook_generatorpy)
    - 13.1 AST-Driven Generation Philosophy
    - 13.2 Parameter Splitting & Widget Values
    - 13.3 Safe Literal Detection
    - 13.4 Cell Construction & Validation

---

## 1. Architecture at a Glance

The backend is organised into **seven layers**, each with a single responsibility:

```
┌──────────────────────────────────────────────────────────────────┐
│  server_fastapi.py         — HTTP entry point (FastAPI)          │
│  serving_api.py            — Serving HTTP routes                 │
│  training_api.py           — Training HTTP routes                │
├──────────────────────────────────────────────────────────────────┤
│  serving_core.py           — Deployment & serving domain logic   │
│  training_core.py          — Training domain logic               │
│  training_datasets.py      — Dataset parsing & storage           │
├──────────────────────────────────────────────────────────────────┤
│  monitoring_core.py        — Metrics aggregation (log-driven)    │
│  cross_validation.py       — K-fold CV helpers                   │
│  training_metrics.py       — Pure-Python metric references       │
├──────────────────────────────────────────────────────────────────┤
│  model_catalog.py          — AST-based model discovery           │
├──────────────────────────────────────────────────────────────────┤
│  server_core.py            — Shared domain: sandbox, sessions,   │
│                              loaders, metrics, HTML              │
├──────────────────────────────────────────────────────────────────┤
│  notebook_generator.py     — Jupyter notebook generation         │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Domain logic is FastAPI-independent.** `server_core.py`, `serving_core.py`, `training_core.py`, `monitoring_core.py`, `cross_validation.py`, and `training_metrics.py` import no FastAPI. The HTTP layers (`server_fastapi.py`, `serving_api.py`, `training_api.py`) are the only modules that touch FastAPI. This makes the domain layer unit-testable in isolation.
- **All model execution happens in a subprocess.** Neither playground, invoke, load, serving prediction, nor training ever imports model code into the server process. The sandbox prelude (`_SANDBOX_PRELUDE`) plus bootstrap strings define what the child can do.
- **State is persisted to JSON files.** `catalog.json`, `deployments.json`, `events.jsonl`, `jobs.json`, `datasets.json` are the source of truth. The in-memory structures (`CatalogStore._catalog`, `ServingStore._models`, `TrainingStore._jobs`, `DatasetStore._items`) are caches that are rehydrated at startup.
- **Monitoring is log-driven, not synthetic.** `monitoring_core.py` computes every metric from events recorded by `ServingStore`. No metric is invented; an empty deployment produces zeros.
- **Training reuses the invocation sandbox.** `training_core.py` calls `core.run_invoke()` inside a `ThreadPoolExecutor` worker. A fit job builds the model constructor, calls the fitting method, and pickles the instance via `run_invoke(..., save_path=...)`. An evaluate job loads the artifact back through the model's own loader and runs a scoring method.

---

## 2. AST-Based Model Catalog (`model_catalog.py`)

### The Core Idea

The catalogue is built by **parsing Python source files with `ast`**, not by importing them. This means:

- **No dependency on numpy/tensorflow/torch.** The catalogue can be built on any machine, even one without the ML stack installed.
- **The model list comes from the package layout, not from hand-written JSON.** Categories are inferred from the filesystem: top-level modules under `likelihood/models/` are categories; sub-packages are sub-categories.
- **Signatures and parameters come from `inspect`-like AST parsing.** `ast.parse` extracts class definitions, method names, parameter names and types, defaults, and docstrings.

### Category Discovery

`build_catalog()` walks the `likelihood/models/` directory tree. For each `.py` file:

1. Parse the file with `ast.parse(source, filename=path)`.
2. Find all top-level `ast.ClassDef` nodes.
3. For each class, collect:
   - **`methods`** — all public methods (including `__init__` parameters as the model's constructor signature).
   - **`parameters`** — the `__init__` parameter names and their annotated types, mapped to widget kinds (`int`, `float`, `bool`, `str`, or `unknown`).
   - **`bases`** — the inheritance chain (for Keras adapter detection in training).
   - **`description`** — the docstring, cleaned of docstring delimiters.
4. Categorise: the category is the parent directory name. A file at `likelihood/models/arima.py` → category `"arima"`; a file at `likelihood/models/deep/lstm.py` → category `"deep"`.
5. Produce a flat list of model dicts: `{id, name, module, signature, category, methods, parameters, description}`.

### `CATEGORY_META` & `DEFAULT_CATEGORY`

A dictionary maps category names to human-friendly labels (Spanish). `DEFAULT_CATEGORY = "general"` is the fallback when a model has no explicit category. The CLI (`--list`, `--json`) uses this for display.

### Why AST Instead of Import?

- **Zero side effects.** Importing `likelihood.models.arima` would execute `from likelihood import ...` at the top of the module, which imports numpy, tensorflow, etc. AST parsing reads only the text.
- **Stability.** The catalogue never crashes because a model class has a syntax error at import time but valid AST.
- **Determinism.** The same source files always produce the same catalogue, regardless of environment.

---

## 3. Core Domain Layer (`server_core.py`)

This is the largest file (4,931 lines) and the shared foundation for everything else.

### 3.1 Catalog Store & Refresh

`CatalogStore` holds the in-memory catalogue and knows where `catalog.json` lives.

- **`changed()`** — returns `True` if the catalogue has been modified since the last persist (dirty flag).
- **`refresh(force=False)`** — re-reads the catalog from disk. If `force` is `True`, bypasses the cache. On success, updates `self._catalog` and the module-level aliases `CATALOG`, `MODELS_BY_ID`. Logs `"Catálogo recargado desde %s: %d modelos"`.
- **Module-level aliases.** `CATALOG_STORE`, `CATALOG`, `MODELS_BY_ID` are set once at module import time and point to the store instance. This avoids circular imports: other modules can `from server_core import CATALOG, MODELS_BY_ID` without importing `CatalogStore` itself.

**Design note:** The catalog is a **single source of truth** for model identity. `training_core.py` resolves every model lookup through `core.MODELS_BY_ID.get(...)`, raising `ModelNotFound` if absent. `serving_core.py` resolves the catalog model for API docs through `find_model(settings.catalog, ...)`. No module maintains its own model registry.

### 3.2 Metrics & Prometheus Counters

The `Metrics` class tracks:

- **Per-endpoint counters** — `requests_total`, `success_total`, `error_total`, `latency_sum_ms`, `latency_max_ms`.
- **Latency percentiles** — P50, P90, P95, P99 computed via a running list (sorted insertion or periodic re-sort).
- **`record(endpoint, status, duration)`** — folds unknown `endpoint` labels into `"other"` so the metrics object never grows unboundedly.

**Prometheus integration:** A `PrometheusMiddleware` in `server_fastapi.py` increments counters for each HTTP response. The middleware is added at app creation, so every request (including 404s and auth failures) is counted.

### 3.3 Parameter Type Classification

`server_core.py` defines constants that map parameter names to their semantic type:

- **`COUNT_PARAMETERS`** — `{n, p, order, max_order, ...}`. These are integer parameters that control the size of a model component.
- **`LENGTH_PARAMETERS`** — `{length, horizon, n_steps, ...}`. Integer parameters that control the length of a sequence.
- **`WIDTH_PARAMETERS`** — `{hidden, units, dim, ...}`. Integer parameters that control the width of a layer.
- **`SHAPE_ARGUMENTS`** — `{shape, dim_shape, ...}`. Parameters that expect a tuple/list shape.
- **`MULTI_SEQUENCE_ARGUMENTS`** — `{x, y, X, Y, inputs, ...}`. Parameters that expect 2D/3D arrays.
- **`SCALAR_INT_ARGUMENTS`** — `{epochs, batch_size, steps, ...}`. Integer scalars with a clear numeric meaning.
- **`SCALAR_FLOAT_ARGUMENTS`** — `{lr, alpha, gamma, ...}`. Float scalars.
- **`SCALAR_FLAG_ARGUMENTS`** — `{shuffle, verbose, ...}`. Boolean flags.

These are used by `method_fields()` to infer the **widget kind** (slider, dropdown, text, toggle) and the **JSON-serialisability** of each parameter. A `SHAPE_ARGUMENTS` parameter is not JSON-serialisable (it expects a tuple), so the UI disables it in the invoke form.

### 3.4 Loader Resolution

`_resolve_loader(model)` determines **how to load a saved model artifact** back into memory. The algorithm:

1. Look for an `artifact.json` in the model's directory.
2. Parse it and find the `loader` field: a dict with `{name, path, path_parameter, requires_instance, module}`.
3. If the loader references a parameter (e.g., `path`), resolve it against the model's directory.
4. Return the resolved loader dict.

`_load_candidates(model)` generates a list of candidate load paths, trying:
- The model's own `load()` or `load_model()` method.
- Keras `load_model` if the model is a Keras `Model` subclass.
- `pickle` fallback.

`_artifact_load_target(model, path)` resolves an artifact folder to the exact file the loader expects. This handles the case where the artifact is a directory (e.g., a Keras HDF5 model) vs. a single pickle file.

### 3.5 Sandboxed Subprocess Execution

This is the **most critical security mechanism** in the backend.

#### Why a Subprocess?

Model code is arbitrary Python. It could import `os`, `subprocess`, `socket`, or perform any other dangerous operation. Running it in the server process would be catastrophic. Instead, every model execution happens in a **child Python process** that:

1. **Has resource limits** set via `resource.setrlimit`:
   - `RLIMIT_CPU` — max CPU seconds.
   - `RLIMIT_AS` — max addressable memory.
   - `RLIMIT_DATA` — max data segment size.
   - `RLIMIT_NOFILE` — max file descriptors.
   - `RLIMIT_FSIZE` — max file size the process can write.

2. **Blocks dangerous imports** via a `guarded_import` hook installed on `sys.meta_path`. Blocked modules include `os`, `sys`, `subprocess`, `socket`, `multiprocessing`, `threading`, `signal`, `ctypes`, `importlib`, `importlib.machinery`, `pkgutil`, `pathlib`, and any module that could access the network or filesystem outside the artifact directory.

3. **Detects catalog-frame execution.** The prelude checks whether the code is being run as part of the catalog build (AST parsing) and skips the sandbox restrictions in that case.

#### The `_SANDBOX_PRELUDE`

A large Python string that is prepended to every subprocess script. It defines:

- **`_emit(event, data)`** — writes a JSON line to stdout, tagged with a marker. The parent process reads stdout line-by-line and extracts the last `_emit` payload as the result.
- **`_limit(name, value)`** — wraps `resource.setrlimit` in a `try/except` so a failed limit doesn't crash the sandbox.
- **`guarded_import`** — the import hook that blocks the listed modules.
- **`_describe(model)`** — introspects a model object for the response payload (class name, method list, attribute values).
- **`_convert_all(value)`** — recursively converts numpy arrays, Keras tensors, and other non-JSON-serialisable objects into plain Python lists/dicts/numbers.
- **`_apply_compile(model, compile)`** — sets the Keras model's `optimizer`, `loss`, `metrics` from the compile spec before fitting.

#### Bootstrap Strings

Each operation gets a **bootstrap string** appended to the prelude. The bootstrap defines the main logic:

- **`_PLAYGROUND_BOOTSTRAP`** — builds the model from the constructor, optionally calls a method, and `_emit`s the result.
- **`_LOAD_BOOTSTRAP`** — loads a model from disk via the resolved loader and `_emit`s its description.
- **`_INVOKE_BOOTSTRAP`** — the workhorse. It can:
  - Build a model from the constructor.
  - Optionally load an artifact first.
  - Call an arbitrary method with the given arguments.
  - Optionally `save_path` to pickle the instance after the call (used by training fit jobs).
  - `_emit` the result (or error).
- **`_SESSION_BOOTSTRAP`** — a **long-lived worker**. It builds the model once, then reads JSON lines from stdin. Each line is `{"id": N, "method": "predict", "arguments": {...}}`. The worker calls the method, writes `{"id": N, "ok": true, "result": ...}` or `{"id": N, "ok": false, "error": "..."}` to stdout. Supports `close` (shuts down), `ping` (health check), `build` (rebuild), and arbitrary method invocations. On error, it catches the exception, writes a traceback, and continues (error recovery).

### 3.6 `run_invoke()` & `InvocationError`

`run_invoke(model, method, parameters, arguments, timeout, load, limits, save_path)` is the **one-shot execution path**.

1. Constructs a Python script from `_SANDBOX_PRELUDE + _INVOKE_BOOTSTRAP + model serialization`.
2. Spawns a `subprocess.Popen` with the resource limits.
3. Waits for the process to finish (with `timeout`).
4. On `subprocess.TimeoutExpired`, raises `InvocationError` with status 504.
5. Parses stdout for the `_marker_payload` (the last `_emit` JSON line).
6. On failure, falls back to the **stderr tail** for Keras progress/memory errors (Keras prints progress to stderr; the actual result is on stdout).
7. Returns a dict: `{status, result, error, elapsed}`.

### 3.7 `ModelSession` — Persistent Workers

`ModelSession` manages a **long-lived subprocess** that keeps the model alive between calls.

- **`wait_ready()`** — polls the subprocess stdout until it reads a `ready` or `error` event. Returns `True` if ready, raises `InvocationError` if error or process died.
- **`invoke(method, arguments, timeout)`** — sends `{"id": N, "method": ..., "arguments": ...}` to stdin, reads the corresponding `{"id": N, "ok": ..., "result"/"error": ...}` from stdout. Tracks `_next_id` for request/response correlation. On timeout or dead worker, sets `_closed=True` and raises `InvocationError`.
- **`close()`** — sends `{"method": "close"}`, waits for the shutdown event, kills the process if it doesn't terminate.
- **`errors` counter** — incremented on each failed invoke. Used by the UI to display "N errors in session".

**Why persistent?** Building a Keras model (loading weights, compiling) is expensive. A one-shot `run_invoke` would rebuild it on every prediction. `ModelSession` amortises the build cost across many invocations. The session is **stateful**: `fit` followed by `predict` sees the trained weights.

### 3.8 `JobQueue` — Background Execution

`JobQueue` wraps a `ThreadPoolExecutor` for background playground runs.

- **Lazy initialisation** — the `ThreadPoolExecutor` is created on the first `submit()`, not at import time.
- **`_prune()`** — called before every `submit()`. Drops expired jobs (finished or older than `max_age`) and enforces the `maximum` cap. This prevents unbounded memory growth from orphaned job objects.
- **`submit(runs)`** — submits a batch of playground runs.
- **`submit_invoke(model, method, parameters, arguments, load, ...)`** — submits a single invocation. Returns a future.

**Design note:** The queue is **in-memory only**. If the server restarts, all queued jobs are lost. This is acceptable because playground runs are ephemeral and idempotent.

### 3.9 HTML Rendering Helpers

`server_core.py` provides helpers to render model information as HTML:

- **`methods_html(model)`** — renders a table of methods with signatures.
- **`attributes_html(model)`** — renders a table of instance attributes.
- **`attributes_section_html(model)`** — wraps `attributes_html` in a `<section>`.
- **`related_html(entries)`** — renders links to related models.

These are used by the frontend to display model details without the frontend needing to know about the model structure.

---

## 4. FastAPI Server (`server_fastapi.py`)

### 4.1 `ServerSettings` & Environment Flags

`ServerSettings` centralises all configuration:

- **`sessions_enabled`** — from `core.SESSIONS_ENABLED` (env var `LM_SESSIONS`).
- **`serving_enabled`** — from `core._env_flag("LM_SERVING", True)`.
- **`api_token`** — from `core.API_TOKEN` (env var `LM_API_TOKEN`). If set, all endpoints require `Authorization: Bearer <token>`.
- **`cors_origins`** — from `core.CORS_ORIGINS` (comma-separated env var). Defaults to `[]` (no CORS).
- **`gzip_min_bytes`** — from `core.GZIP_MIN_BYTES`. Responses below this size are not gzipped.
- **`trust_proxy`** — from `core.TRUST_PROXY`. When `False`, the server uses the direct client IP for rate limiting; when `True`, it trusts the `X-Forwarded-For` header.
- **`template`** — initial `""`, lazy-loaded via `preview_template()`.

**`refresh()`** — returns early if the catalog was explicitly injected (tests use `--package-dir` to inject a catalogue). Otherwise, calls `self.store.refresh()` and updates `self.catalog`.

### 4.2 `_AliasedBody` & Legacy Compatibility

`_AliasedBody(BaseModel)` uses `ConfigDict(extra="allow")` to **tolerate legacy field names**. The `payload()` method:

1. Calls `model_dump()` to get the validated fields.
2. Drops `None` values (so optional fields don't override defaults).
3. Merges `model_extra` (the unknown fields) into the result.

This means an old client can send `{"id": "arima", "params": {...}}` and the server will accept `id` as an alias for `model_id` and `params` as an alias for `parameters`, while new clients use the canonical names.

### 4.3 Endpoint Routing

`server_fastapi.py` defines the following route groups:

- **`GET /api/models`** — list all models with pagination, search, category filter.
- **`GET /api/models/{model_id}`** — full model detail (methods, parameters, description).
- **`GET /api/models/{model_id}/versions`** — version history.
- **`POST /api/models/{model_id}/invoke`** — one-shot method invocation.
- **`POST /api/models/{model_id}/load`** — load an artifact and return its description.
- **`GET /api/models/{model_id}/files`** — list artifact files.
- **`POST /api/models/{model_id}/sessions`** — create a persistent session.
- **`DELETE /api/sessions/{session_id}`** — close a session.
- **`GET /api/sessions/{session_id}`** — session status.
- **`POST /api/sessions/{session_id}/invoke`** — invoke a method on the session.
- **`GET /api/models/{model_id}/notebook`** — generate a Jupyter notebook.
- **`GET /health`** — health check.
- **`GET /metrics`** — Prometheus metrics.

### 4.4 CORS, Gzip, Proxy Trust

- **CORS** — `CORSMiddleware` is added only if `cors_origins` is non-empty.
- **Gzip** — `GZipMiddleware` is added with `minimum_size=gzip_min_bytes`.
- **Proxy trust** — the rate limiter and auth middleware use `request.client.host` if `trust_proxy` is `False`, or parse `X-Forwarded-For` if `True`.

### 4.5 Prometheus Middleware & Rate Limiting

- **Prometheus middleware** — wraps every request. Increments `http_requests_total` (by method, status, endpoint label) and records `http_request_duration_seconds`.
- **Rate limiter** — a `TokenBucket` middleware. Each client IP gets a bucket with `rate` tokens/sec and `burst` capacity. On 429, the response body is a JSON error. The bucket is refilled over time.

### 4.6 Error Handling & Authentication

- **`APIError`** — a base exception with `status` and `detail`. Handled by a FastAPI exception handler that returns a JSON `{status, message}`.
- **`ValidationError`** — subclass of `APIError` with status 400.
- **`ModelNotFound`** — subclass of `APIError` with status 404.
- **`AuthError`** — status 401. Raised by the auth middleware if the token is missing or invalid.
- **`InvocationError`** — status 500 or 504. Carries `detail` (the traceback).

**Authentication middleware:** If `api_token` is set, every request (except `GET /health`) must include `Authorization: Bearer <token>`. The middleware compares the token against `settings.api_token` using `hmac.compare_digest` (constant-time comparison to prevent timing attacks).

---

## 5. Serving Core (`serving_core.py`)

### 5.1 `ServingStore` & Concurrency Model

`ServingStore` manages **deployments** — persistent model endpoints.

- **`__init__`** — initialises:
  - `STORE_DIR` (default `data/store/`).
  - `deployments_path` = `STORE_DIR / "deployments.json"`.
  - `models_path` = `STORE_DIR / "models.json"`.
  - `threading.RLock` — a reentrant lock. Reentrant because `deploy()` calls `_persist_locked()` which acquires the lock, and `_persist_locked` may be called from within `_record_error` which is called from within `deploy()` which already holds the lock.
  - **Per-endpoint `BoundedSemaphore`** — `self._semaphores[endpoint_id]`. This ensures that only **one load/deploy runs concurrently per endpoint**. If a deploy is in progress, a second deploy request for the same endpoint blocks (or times out) rather than corrupting the deployment.
  - **`ThreadPoolExecutor(DEPLOY_WORKERS)`** — manages parallelism **across** endpoints. The semaphore is per-endpoint; the pool is global.
  - **`EventLog`** — the append-only JSON lines log.
  - **`_load_models()`** — reads `models.json` and normalises each record.

- **`_load_models()`** — reads `models.json`, calls `_normalise_model_record()` for each record, stores in `self._models` (a list of dicts).

### 5.2 `EventLog` — Append-Only JSON Lines

`EventLog` appends JSON lines to `events.jsonl`.

- **`record(event)`** — appends one JSON line. Thread-safe (uses the `RLock`).
- **`_rewrite_locked()`** — compacts the log: reads all lines, drops events older than `MAX_EVENTS`, writes them back atomically (write to temp file, `os.replace`).
- **`_migrate_event_log()`** — schema migration. Reads all lines, adds `model_instance_id` and `catalog_model_id` fields to old events that have `model_id` but no `model_instance_id`, writes them back. This ensures backward compatibility with old event schemas.

**Why append-only JSON lines?** It's the simplest format for concurrent appenders. Each event is one line, so no two writers can interleave a partial JSON object. Compaction (`_rewrite_locked`) handles the unbounded growth.

### 5.3 Event Schema

Each event has:

- `deployment_id` — the deployment this event belongs to.
- `endpoint_id` — the endpoint (same as `deployment_id` in this context).
- `endpoint_name` — human-friendly name.
- `model_id` — the catalog model ID.
- `model_instance_id` — the specific instance (for group tracking).
- `model_type_id` — the model type (for grouping).
- `catalog_model_id` — redundant with `model_id` but kept for the migration path.
- `model_name` — human-friendly name.
- `method` — the method that was invoked.
- `status` — `"ok"` or `"error"`.
- `http_status` — the HTTP status code (200, 500, etc.).
- `latency_ms` — rounded to 2 decimal places.
- `source` — `"serving"`.
- `request_bytes` / `response_bytes` — size of the request/response bodies.
- `input_preview` / `input_summary` / `output_preview` — truncated previews of the data.
- `error_type` / `error_message` / `error_detail` — for error events.

### 5.4 Model Record Normalisation & Grouping

`_normalise_model_record()` normalises a raw model record from `models.json`:

- Ensures `model_id`, `name`, `module`, `signature`, `category` are present.
- Normalises `signature` (a string like `"__init__(self, n, p, ...)"`).
- Ensures `category` defaults to `DEFAULT_CATEGORY`.

`_model_records()` groups records by `signature`:

- For each group, counts **active deployments** (status `"running"`).
- Returns the group with the most active deployments, or the most recently updated one if there are no active deployments.
- Sorts by `created_at` descending.

**Why group by signature?** Multiple deployments can use the same model signature (e.g., two `Arima` instances with different data). Grouping lets the UI show "This model has N active endpoints" without listing each one separately.

### 5.5 Deployment Lifecycle

`deploy(settings, deployment)` is the main entry point:

1. **Resolve the model path.** The deployment specifies a `model_type_id` and a `path` (or `path_parameter`). The algorithm:
   - Look up the catalog model by `model_type_id`.
   - Resolve the loader via `core.resolve_loader(model)`.
   - Resolve the actual file path from the artifact folder or the model's basename.
   - Log the resolution via `_log()`.

2. **Persist before executing.** Calls `_persist_locked()` to write the deployment to `deployments.json` **before** running the load. This means if the load crashes, the deployment record still exists and the UI can show it as `"pending"` or `"failed"`.

3. **Execute the load.** Calls `core.run_load(model, prepared, timeout=DEPLOY_TIMEOUT)`. This spawns a sandboxed subprocess that loads the model and `_emit`s its description.

4. **Construct the artifact dict.** On success, the artifact captures:
   - `loader` — the name of the loader that was used.
   - `path` — the resolved file path.
   - `type` — the model class name.
   - `module` — the Python module.
   - `repr` — `repr(model)` from the sandbox.
   - `attributes` — list of attribute names.
   - `attribute_values` — dict of attribute values.
   - `attribute_keys` — dict of attribute keys (for JSON serialisation).
   - `elapsed` — time taken to load.
   - `loaded_at` — ISO timestamp.

5. **Set status to `"running"`.** Update `deployed_at` and `loaded_artifact`.

6. **Apply sample hints.** `apply_sample_hints(method_fields, loaded_artifact)` updates the `method_fields` to reflect the actual model attributes (e.g., if the model has a `n` attribute set to 5, the hint for the `n` parameter is updated).

7. **Clear `last_error`.** If the previous load failed, the error is cleared on success.

### 5.6 Error Recording

`_record_error(deployment, event)` builds an error event:

- `request_bytes` — the size of the request body (from the HTTP layer).
- `response_bytes` — 0 (no response was sent).
- `input_preview` — truncated request body.
- `input_summary` — a summary of the request fields.
- `output_preview` — empty (no output).
- `error_type` — default `"Error"`, overridable.
- `error_message` — truncated to 1000 chars.
- `error_detail` — last 4000 chars of the traceback.
- Records via `self.events.record(event)`.

### 5.7 `events_for()`

`events_for(endpoint_id, limit=100)` filters events by `endpoint_id` and returns up to `max(limit*8, limit)` recent events. The factor of 8 accounts for the fact that error events have more fields than success events, so a fixed number of lines doesn't give a fixed number of useful events.

### 5.8 API Documentation Generation

`serving_core.py` generates **complete API documentation** programmatically:

- **`payload_example`** — a JSON example of the request body.
- **`headers_example`** — a JSON example of the headers.
- **`samples`** — code samples in Python, bash/cURL, JavaScript, and raw HTTP.
- **`request_schema`** — the JSON Schema for the request. Includes `input`/`inputs`/`arguments` envelope, `fields` with `name`, `type`, `required`, `default`, `description`, `example`, `jsonable`, and `headers`.
- **`response_schema`** — the JSON Schema for the response. Includes `success` (boolean), `result` (any), `error` (string), `status` (integer).
- **`model` info** — `{id, model_type_id, name, title, module, signature, category, category_label}`.
- **`method` info** — `{description, kind, inference_like, inference_hint}`.

**Why generate docs programmatically?** The docs are **guaranteed to match the actual behaviour**. If you add a new field to the request schema, the docs update automatically. No hand-written markdown to drift.

---

## 6. Serving API (`serving_api.py`)

### 6.1 Deployment Endpoints

- **`POST /api/serving/deploy`** — creates a deployment. Body: `{model_id, method, path, ...}`. Calls `serving_core.deploy()`.
- **`DELETE /api/serving/deployments/{id}`** — undeploys. Sets status to `"stopped"`.
- **`GET /api/serving/deployments`** — lists all deployments with per-endpoint stats.
- **`GET /api/serving/deployments/{id}`** — deployment detail with artifact info.
- **`GET /api/serving/endpoints/{id}`** — endpoint detail with model info, method info, and API docs.

### 6.2 Predict Endpoint

- **`POST /api/serving/endpoints/{id}/predict`** — invokes the deployed model. Body: `{input, inputs, arguments, fields}`. The algorithm:
  1. Look up the deployment by ID.
  2. Verify status is `"running"`.
  3. Resolve the method via `core.find_method(model, deployment["method"])`.
  4. Get fields via `core.method_fields(method, parameters)`.
  5. Apply `apply_sample_hints()`.
  6. Call `core.run_invoke()` with the model, method, and arguments.
  7. Record the event (success or error) via `ServingStore`.
  8. Return the result or the error.

### 6.3 Monitoring & Events Endpoints

- **`GET /api/serving/monitoring/overview`** — overall stats: total requests, success rate, avg latency, P95 latency, error rate.
- **`GET /api/serving/monitoring/deployments`** — per-deployment stats.
- **`GET /api/serving/monitoring/events`** — recent events, filterable by deployment, status, time range.
- **`GET /api/serving/monitoring/timeseries`** — time-bucketed metrics for charting.

### 6.4 `deployment_payload()` & Per-Endpoint Stats

`deployment_payload(settings, deployment, request, stats=None)` constructs the full endpoint payload:

- Copies the deployment dict, pops the internal `_method` key.
- Resolves the model via `find_model(settings.catalog, ...)`.
- Finds the method via `core.find_method(model, deployment["method"])`.
- Gets fields via `core.method_fields(method, ...)`.
- Applies `apply_sample_hints()`.
- Constructs the predict URL: `f"/api/serving/endpoints/{deployment['id']}/predict"` from `request.base_url`.
- Gets `label, tone` from `serving_core.STATUS_LABELS` for UI theming.

`_stats_by_endpoint(events)` is a **single-pass aggregation**: groups all events by `endpoint_id` in one loop, then calls `monitoring_core.aggregate(group)` for each group and adds `last_request_at`. This avoids multiple passes over the event list.

---

## 7. Training Core (`training_core.py`)

### 7.1 Training Architecture & Job Types

Two job types:

- **`fit`** — builds the model constructor, calls the fitting method in a sandbox, and the child pickles the instance via `run_invoke(..., save_path=...)`. The artifact is written to `STORE_DIR/artifacts/<artifact_id>/`.
- **`evaluate`** — loads the artifact (or builds a fresh model), runs a scoring method. No artifact is written.

**Stateless at process level:** A `ThreadPoolExecutor` manages background workers. If the server restarts, running jobs are marked `failed` (a worker that dies with the process cannot resume a fitting loop).

### 7.2 `TrainingStore` & Persistence

`TrainingStore` manages jobs:

- **`__init__`** — loads `jobs.json` from `STORE_DIR`. Each job is a dict with `job_id`, `kind`, `status`, `model_id`, `method`, `parameters`, `arguments`, `compile`, `control`, `cross_validation`, `normalization`, `target_normalization`, `timeout_seconds`, `created_at`, `started_at`, `finished_at`, `elapsed`, `error`, `error_detail`, `artifact`, `log`.
- **`_persist_locked()`** — atomically writes `jobs.json` (write to temp file, `os.replace`).
- **`submit(spec)`** — validates the spec, creates a job dict, appends to `self._jobs`, persists, and submits to the `ThreadPoolExecutor`.
- **`job(job_id)`** — looks up a job by ID.
- **`jobs()`** — returns all jobs, sorted by `created_at` descending.

### 7.3 Framework Fit Detection (Keras Adapter)

**The problem:** `model_catalog.py` only discovers methods defined **inside** the `likelihood` package. If a model class inherits from `keras.Model`, the `fit()` method is defined in `keras`, not in `likelihood`, so it **never appears** in `model["methods"]`. The UI would think the model can't be trained.

**The solution:** `supports_framework_fit(model)` checks:

1. The model's `bases` set intersects `FRAMEWORK_MODEL_BASES` (e.g., `{"keras.Model", "tensorflow.keras.Model"}`).
2. The model's `module` starts with `"likelihood.models"`.

If both are true, `framework_fit_method(model)` returns a **synthetic catalogue entry**:

```python
{
    "name": "fit",
    "signature": "fit(x, y, epochs=1, batch_size=32, validation_split=0.0, verbose=0, shuffle=True)",
    "description": "Entrena el modelo Keras con el compilador por defecto...",
    "parameters": [dict(p) for p in FRAMEWORK_FIT_PARAMETERS],
    "kind": "method",
    "decorators": [],
    "inherited": False,
    "framework": "keras",
}
```

`training_fit_method(model)` resolution order:

1. Iterate `FIT_METHODS` (a list like `["fit", "train", "fit_predict"]`) and find the first one that exists in `model["methods"]` via `core.find_method(model, name)`.
2. Only if none is found, check `supports_framework_fit(model)` and return the synthetic Keras adapter.

**Why not modify the package?** The adapter is a **read-only overlay**. The actual `likelihood` source is never modified. The synthetic method is only visible to the training layer, not to the catalog or the playground.

### 7.4 Job Submission & Normalisation

`submit(spec)` validates and normalises the spec:

- **`model_id`** — required. Resolved via `_find_model()`.
- **`kind`** — `"fit"` or `"evaluate"`. Defaults to `"fit"`.
- **`method`** — the fitting method name. Defaults to the first fitting method found.
- **`parameters`** — the constructor parameters (dict).
- **`arguments`** — the method arguments (dict).
- **`compile`** — Keras compile spec (optimizer, loss, metrics). Only for `fit`.
- **`control`** — metric control spec (target metric, threshold). Only for Keras `fit`.
- **`cross_validation`** — CV spec (folds, seed, metric, mode). Only for Keras `fit`.
- **`normalization`** — `"none"`, `"standard"`, or `"minmax"`. Applied to the target variable.
- **`target_normalization`** — per-target normalisation (for multi-output models).
- **`dataset`** — dataset reference (ID + columns).
- **`timeout_seconds`** — max execution time. Defaults to `TRAIN_TIMEOUT`.

**Cross-validation validation:**

- Requires `dataset_meta` with `mode == "xy"` (X + Y columns). Raises `ValidationError` otherwise.
- `cross_validation.normalise_options(raw_cv)` normalises the CV options.
- Checks `rows < cv_options["folds"]` → raises `ValidationError`.
- Rejects combining `cross_validation` with `arguments.validation_data` (conflict).
- Requires `fit` context: `control` or `cross_validation` without a fit raises `ValidationError`.

### 7.5 Cross-Validation Integration

When `cross_validation` is specified:

1. `cross_validation.normalise_options(raw_cv)` returns `{folds, seed, metric, mode}`.
2. The data is split into `folds` folds using `seed` for reproducibility.
3. For each fold:
   - Train the model on the training split.
   - Score on the validation split.
   - Record the fold score.
4. Select the best fold using `cross_validation.is_better(score, best_score, mode)` where `mode` is `"minimise"` or `"maximise"`.
5. Log per-fold info and warnings in Spanish for missing metrics.

### 7.6 Artifact Management & Atomic Swap

After a fit job completes, the artifact is written to `STORE_DIR/artifacts/<artifact_id>/`:

- The sandbox writes the model file (Keras HDF5, pickle, etc.).
- `training_core.py` adds `artifact.json` with metadata.
- Sets `load_path` for serving.

**Atomic swap:**

1. Write `artifact.json` to a staging directory.
2. Back up the old artifact dir to `.{artifact_id}-old-{uuid_hex[:8]}`.
3. `os.replace(staging, artifact_dir)` — atomic on POSIX.
4. Clean up the backup.
5. On any `BaseException`: if the target dir is missing, restore the backup; clean up staging; re-raise.

**Why atomic?** A partial write (e.g., a Keras model file that's 50% written) would leave a corrupt artifact. The atomic swap ensures that the artifact is either the old one or the new one, never a hybrid.

### 7.7 `_run_fit` — Full Training Pipeline

`_run_fit(job)` is the worker function:

1. Resolve the model via `_find_model(job["model_id"])`.
2. Resolve the fitting method via `training_fit_method(model)`.
3. If `cross_validation` is specified, split the data and run CV.
4. Build the sandbox script:
   - `_SANDBOX_PRELUDE + _INVOKE_BOOTSTRAP + model serialization`.
   - The bootstrap builds the model, calls the fit method, and pickles the instance to `save_path`.
5. Call `core.run_invoke()` with the appropriate limits.
6. On success:
   - Copy the artifact files to `STORE_DIR/artifacts/<artifact_id>/`.
   - Write `artifact.json`.
   - Update the job: `status="done"`, `artifact` populated, `finished_at`, `elapsed`.
7. On failure:
   - Update the job: `status="failed"`, `error`, `error_detail`, `finished_at`, `elapsed`.
   - Append to `job["log"]`.

### 7.8 `_run_evaluate` — Scoring Pipeline

`_run_evaluate(job)`:

1. Resolve the model.
2. If `artifact` is specified and `save_method == "pickle-batch"`, raise `ValidationError` (batch artifacts are only evaluable via the series analysis path).
3. If an artifact path exists:
   - Resolve the loader via `core.resolve_loader(model)`.
   - Build a `load` dict: `{name, path, path_parameter, requires_instance, module}`.
4. Find the scoring method via `core.find_method(model, job["method"])`.
5. Call `core.run_invoke()` with `load` and `limits={"cpu": TRAIN_CPU_SECONDS}`.
6. Return the result.

**The artifact is optional:** If no artifact is given, the model is built from the constructor and the scoring method runs on that fresh instance. This is useful for testing a model on a small dataset without a prior fit.

### 7.9 Series Analysis & Overwrite

The series analysis endpoint (`POST /api/training/series`) takes a time-series dataset and a model, fits the model, evaluates it on a validation split, and returns the metrics.

**Overwrite:** If the job's `artifact_id` matches an existing artifact, the new fit overwrites the old one:

- The old artifact dir is backed up.
- The new artifact is written.
- The job log records `"Modelo sobrescrito con la configuración de Analizar series: escalado {scaling}, argumentos {fit_arguments}"`.
- The return message is `"Modelo sobrescrito con la configuración analizada"`.

**`_series_metrics(actual, predicted)`:** Pairs finite (truth, forecast) via `zip`, computes RMSE, MAE, MAPE. Skips non-finite pairs. MAPE excludes zero-truth pairs.

### 7.10 Error Handling in Job Execution

The job execution is wrapped in a `try/except`:

- **`APIError`** — sets `job["status"] = "failed"`, `job["finished_at"]`, `job["elapsed"]`, `job["error"] = str(exc)`, `job["error_detail"] = detail[-8000:]`. Appends to `job["log"]`. Persists.
- **`BaseException`** — catches everything else (including `KeyboardInterrupt` during shutdown). Same treatment as `APIError` but with a different log level.

**Design note:** The full traceback is available in `error_detail` (shown in the error card), but the job log timeline gets only a short `error` entry. A second log entry says "Detalle técnico disponible en la tarjeta de error" to guide the user. This avoids mixing Keras progress output with real log events in the timeline.

### 7.11 `_atomic_write` & `_copy_tree`

**`_atomic_write(path, text)`:**

1. Write to a temp file: `.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp`.
2. `flush()` + `os.fsync()`.
3. `os.replace(temp, path)` — atomic.
4. Creates parent dirs if needed.

**Why process+thread ID?** Two concurrent writers to the same file (e.g., two training jobs finishing at the same time) would collide on a fixed temp name. The PID+thread ID makes the temp name unique.

**`_copy_tree(source, target)`:**

1. `target.mkdir(parents=True, exist_ok=True)`.
2. If `source` doesn't exist, return.
3. For each `entry` in `source.iterdir()`:
   - If `entry.is_dir()`: `shutil.copytree(entry, destination, dirs_exist_ok=True)`.
   - Else: `shutil.copy2(entry, destination)`.

---

## 8. Training API (`training_api.py`)

### 8.1 Router Design & Reuse of `server_core` Errors

The training router is a `FastAPI` `APIRouter` with `tags=["training"]`. It reuses:

- **`server_core.APIError`** hierarchy — so the application error handlers, auth middleware, rate limiter, and Prometheus counters all apply without extra wiring.
- **`get_settings(request)`** — the same dependency as `server_fastapi.py`.
- **`require_training(settings)`** — raises `APIError("La invocación de métodos está deshabilitada en este servidor", status=403)` if `settings.invoke_enabled` is falsy. Training reuses the invocation sandbox, so the invoke switch must be on.

### 8.2 Job Payload & Summary

`_job_payload(settings, job)` constructs the full job payload:

- `job_id`, `kind`, `status`, `name`.
- `model_id`, `model_name` (falls back to `model_id`).
- `method`, `adapter` (the fitting method that was actually used).
- `dataset` — the dataset reference.
- `arguments` (dict copy), `parameters` (dict copy), `compile` (dict copy), `control` (dict copy), `cross_validation` (dict copy).
- `normalization` (string), `target_normalization` (dict copy).
- `timeout_seconds`, `attempts`.
- `created_at`, `started_at`, `finished_at`, `elapsed`, `compute_seconds`.
- `error`, `artifact_id`, `artifact_path`, `artifact_size_bytes`.
- `result_type`, `result_repr`.

**Why dict copies?** The job dict is mutable and shared across the `TrainingStore`. If the API returned a direct reference to `job["arguments"]`, a client modification (via deserialisation and re-serialisation) could corrupt the in-memory state. A fresh copy prevents this.

### 8.3 Series Analysis Request Model

`SeriesAnalysisRequest(BaseModel)`:

- `validation_ratio` — float, default 0.2, `gt=0`, `lt=1`. "Fracción final de la serie reservada para validación (0-1)".
- `n_steps` — int, default 1. "Número de pasos a predecir".
- `columns` — optional dict mapping column names to roles (`"target"`, `"feature"`, etc.).
- `scaling` — optional string (`"none"`, `"standard"`, `"minmax"`).
- `model_id` — the model to use.
- `parameters` — constructor parameters.
- `arguments` — fit method arguments.
- `compile` — Keras compile spec.
- `timeout_seconds` — max execution time.

### 8.4 Dataset Upload & Management

- **`POST /api/training/datasets`** — uploads a CSV/XLSX/Parquet file. Parses it into a `pandas.DataFrame`, stores in `DatasetStore`, returns the dataset metadata.
- **`POST /api/training/datasets/url`** — fetches a dataset from a URL. Validates the URL (no `file://`, no internal IPs), fetches, parses.
- **`GET /api/training/datasets`** — lists all datasets.
- **`GET /api/training/datasets/{id}`** — dataset detail with column metadata and preview.
- **`DELETE /api/training/datasets/{id}`** — deletes the dataset.

---

## 9. Training Datasets (`training_datasets.py`)

### 9.1 Format Detection & Limits

`sUPPORTED_FORMATS = frozenset({"csv", "tsv", "xlsx", "parquet"})`.

`_format_from_name(name)` infers the format from the filename/URL path suffix. Raises `core.ValidationError("El formato .xls no está soportado; guarda el libro como .xlsx")` for `.xls`.

**Limits** (all overridable via env vars):

- `MAX_BYTES` — `LM_TRAINING_DATASET_BYTES` or 25 MB.
- `MAX_ROWS` — `LM_TRAINING_DATASET_ROWS` or 50,000.
- `MAX_COLUMNS` — `LM_TRAINING_DATASET_COLUMNS` or 200.
- `MAX_DATASETS` — `LM_TRAINING_DATASETS` or 20.
- `PREVIEW_ROWS` — 8.
- `SAMPLE_SEED` — 42.

### 9.2 `DatasetStore` & In-Memory Storage

`DatasetStore` keeps parsed DataFrames **in the server process memory**:

- `self._items: Dict[str, Dict]` — maps dataset ID to `{id, name, format, shape, columns, preview, created_at, size_bytes}`.
- **`register(data, meta)`** — stores the DataFrame and metadata. Returns the dataset ID.
- **`get(dataset_id)`** — returns the dataset dict (including the DataFrame).
- **`list()`** — returns all dataset metadata (no DataFrames).
- **`delete(dataset_id)`** — removes the dataset.

**Why in-memory?** A job stores only a compact dataset reference plus the selected feature/target columns. This keeps `jobs.json` small even for the 20k-row housing example. The DataFrame itself is not serialised to JSON; it lives in RAM.

### 9.3 Column Metadata & Preview

`_column_meta(frame)` describes every column for browser selectors:

- `name` — column name.
- `dtype` — dtype string (e.g., `"int64"`, `"float64"`, `"object"`).
- `non_null` — count of non-null values.
- `unique` — count of unique values (via `nunique(dropna=True)`).
- `numeric` — bool, `is_numeric_dtype` and **not** `is_bool_dtype`. This ensures boolean columns are not treated as numeric in the UI.
- `sample` — first non-null value, JSON-safe via `_json_value`.

`_preview(frame)` returns the first `PREVIEW_ROWS` rows as a list of dicts, each value JSON-safe.

### 9.4 Sampling & Time-Series Preservation

**`_sample_rows(data, rows)`:** Returns full data if `rows is None or rows <= 0 or rows >= len(data)`. Otherwise, `data.sample(n=int(rows), random_state=SAMPLE_SEED)`. The docstring notes this keeps every class/region represented even when the file is sorted (the Iris CSV is grouped by species).

**`_limit_series(data, rows)`:** Returns full data if `rows is None or rows <= 0 or rows >= len(data)`. Otherwise, `data.tail(int(rows))`. The docstring: "Keep the most recent ``rows`` values (time order is never shuffled)." This is used for time-series datasets where row order matters.

### 9.5 Remote URL Fetching & IP Validation

When a dataset is fetched from a URL:

1. Parse the URL with `urlparse`.
2. Reject `file://` URLs (no local file access).
3. Resolve the hostname with `socket.getaddrinfo`.
4. Check the resolved IP against `ipaddress` blocks: reject `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`, `127.0.0.0/8`, and other private/reserved ranges.
5. Fetch with `urllib.request.urlopen` with a timeout.
6. Check the content length against `MAX_BYTES`.
7. Parse into a DataFrame with the appropriate `pandas` reader.

**Why IP validation?** A malicious URL could point to an internal service (e.g., `http://169.254.169.254/latest/meta-data/` for AWS instance metadata). The IP validation prevents SSRF (Server-Side Request Forgery) attacks.

---

## 10. Training Metrics (`training_metrics.py`)

### Confusion Matrix

`confusion_matrix(y_true, y_pred, labels=None)`:

- Validates equal lengths.
- Builds a matrix with `positions` dict mapping label → index.
- Ignores out-of-label pairs.
- Returns `(labels, matrix)`.

### Cohen's Kappa

A **pure-Python reference implementation**:

```python
def cohens_kappa(y_true, y_pred):
    labels, matrix = confusion_matrix(y_true, y_pred)
    total = sum(sum(row) for row in matrix)
    if not total:
        return 0.0
    size = len(matrix)
    observed = sum(matrix[i][i] for i in range(size)) / total
    row = [sum(values) for values in matrix]
    column = [sum(matrix[i][j] for i in range(size)) for j in range(size)]
    expected = sum(row[i] * column[i] for i in range(size)) / (total * total)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)
```

**Why pure Python?** The trainable Keras metric lives in the sandbox bootstrap (`_INVOKE_BOOTSTRAP`). The pure-Python version is for **unit tests** and **offline evaluation** — no TensorFlow dependency.

**`_as_labels()`:** Unwraps numpy scalars via `.item()` so that `np.float64(1.0)` becomes `1.0` and `np.int64(3)` becomes `3`. This ensures the confusion matrix works with both raw lists and numpy arrays.

---

## 11. Cross-Validation (`cross_validation.py`)

### Constants

- `DEFAULT_FOLDS = 5`
- `MIN_FOLDS = 2`
- `MAX_FOLDS = 20`
- `DEFAULT_SEED = 42`
- `DEFAULT_METRIC = "val_loss"`
- `MINIMISE_TOKENS` — a frozenset of metric names that should be minimised (`"loss"`, `"val_loss"`, `"rmse"`, `"mae"`, `"mape"`, etc.).

### `infer_mode(metric)`

Returns `"minimise"` if the metric is in `MINIMISE_TOKENS`, else `"maximise"`.

### `normalise_options(raw)`

Normalises a raw CV options dict:

- `folds` — clamped to `[MIN_FOLDS, MAX_FOLDS]`, defaults to `DEFAULT_FOLDS`.
- `seed` — int, defaults to `DEFAULT_SEED`.
- `metric` — string, defaults to `DEFAULT_METRIC`.
- `mode` — string, defaults to `infer_mode(metric)`.

### `is_better(score, best, mode)`

Returns `True` if `score` is better than `best` according to `mode`:

- `"minimise"`: `score < best`.
- `"maximise"`: `score > best`.

This is the single comparison function used by the training layer to select the best fold.

---

## 12. Monitoring Core (`monitoring_core.py`)

### 12.1 Time Range Presets & Buckets

`RANGE_PRESETS` maps short labels to `(label, duration_seconds, bucket_seconds)`:

- `"1h"` → `("Última hora", 3600, 60)`
- `"24h"` → `("Últimas 24 horas", 86400, 900)`
- `"7d"` → `("Últimos 7 días", 604800, 3600)`
- `"30d"` → `("Últimos 30 días", 2592000, 86400)`

`_BUCKETS = (60, 300, 900, 1800, 3600, 10800, 21600, 43200, 86400, 604800)` — the bucket sizes in seconds.

`MAX_RANGE_SECONDS = 366 * 86400` (366 days, to cover a leap year).

`_BUCKET_LABELS` maps each bucket size to a Spanish label (e.g., `60 → "1 minuto"`, `86400 → "1 día"`).

### 12.2 `aggregate()` — The Metrics Engine

`aggregate(events)` iterates over a list of events and collects:

- `latencies` — from `event.get("latency_ms")`.
- `compute` — from `event.get("compute_seconds")`.
- `statuses` — from `str(event.get("http_status") or 0)`.
- `error_types` — from `event.get("error_type")` where `event.get("status") == "error"`.
- `endpoints` — from `event.get("deployment_id") or event.get("endpoint_id")`.
- `models` — from `event.get("model_id") or event.get("model_type_id")`.
- `request_bytes`, `response_bytes`.

Returns a dict:

- `requests` — count.
- `successful` — count where `status == "ok"`.
- `failed` — count where `status == "error"`.
- `success_rate` — `successful / requests * 100`, rounded to 2.
- `error_rate` — `failed / requests * 100`, rounded to 2.
- `avg_latency_ms` — rounded to 2.
- `p50/p90/p95/p99_latency_ms` — via `percentile(latencies, p)`.
- `max_latency_ms`.
- `avg_compute_seconds` — rounded to 3.
- `p50/p95_compute_seconds` — via `percentile(compute, p)`.
- `max_compute_seconds` — rounded to 3.
- `compute_seconds_total` — rounded to 2.
- Plus: `top_error_types`, `top_endpoints`, `top_models`, `last_request_at`.

### 12.3 `range_window()` & `bucketize()`

`range_window(range_key, end)`:

- `range_key` is a preset key (`"1h"`, `"24h"`, etc.) or a custom ISO timestamp.
- `end` is the "now" timestamp (defaults to current time).
- Returns `(start, end)` as ISO strings.

`bucketize(events, start, end, bucket_seconds)`:

- Buckets events into time intervals of `bucket_seconds`.
- Returns a list of `{bucket, start, end, metrics}` where `metrics` is the result of `aggregate(events_in_bucket)`.
- Empty buckets are filled with zeros.

### 12.4 `deployment_overview()`

`deployment_overview(deployments, events)`:

- For each deployment, filters events by `deployment_id`.
- Calls `aggregate()` on the filtered events.
- Returns a dict mapping deployment ID to `{**aggregate(events), "last_request_at": ...}`.

---

## 13. Notebook Generator (`notebook_generator.py`)

### 13.1 AST-Driven Generation Philosophy

**Every notebook is derived from `model_catalog`**, which introspects the real package sources via AST. As a consequence:

- The **model list** comes from the package layout, not from hand-written JSON.
- **Parameter tables and widgets** are built from the real `__init__` signature.
- The **example code** is generated from that same signature and stays executable.
- The documentation can **never drift** from the implementation.

**Validation:** When `nbformat` is installed, notebooks are validated against the official notebook schema before being written. Otherwise, the plain `dict` described by the schema is used.

### 13.2 Parameter Splitting & Widget Values

`_split_parameters(model)`:

- Splits model parameters into `tunable` (widget kind in `("int", "float", "bool", "str")`) and `fixed` (all others, including variadic).
- Returns `(tunable, fixed)`.

**Why split?** The UI shows tunable parameters as interactive widgets (sliders, toggles, text inputs) and fixed parameters as static values. Variadic parameters are skipped because they can't be represented as a single widget.

`_widget_value(parameter)`:

- `bool` → `"True"` or `"False"`.
- `int` → `str(int(float(default)))` with fallback `"1"`.
- `float` → `repr(float(default))` with fallback `"0.0"`.
- `str` → keeps the documented default or a neutral placeholder.

### 13.3 Safe Literal Detection

`_is_safe_literal(text)`:

- Parses with `ast.parse(text, mode="eval").body`.
- Walks the AST.
- Returns `False` if any `ast.Name` node (except `True`/`False`/`None`) or any `ast.Call`/`ast.Attribute`/`ast.Subscript` is found.
- Returns `False` if text is empty or `SyntaxError`.

**Why?** A parameter default like `"likelihood.models.arima"` is a safe string literal. But a default like `"__import__('os').system('rm -rf /')"` would be dangerous if interpolated into generated code. The AST check ensures only pure literals are used.

### 13.4 Cell Construction & Validation

A notebook is built from cells:

- **Markdown cell** — model name, description, category.
- **Code cell** — `from likelihood.models.arima import Arima`.
- **Markdown cell** — parameter table (markdown table).
- **Code cell** — example instantiation with tunable parameters.
- **Markdown cell** — methods table.
- **Code cell** — example method call.

`generate_notebook(model, output_dir)`:

- Builds the cells.
- Creates the notebook dict: `{"cells": [...], "metadata": {...}, "nbformat": 4, "nbformat_minor": 5}`.
- If `nbformat` is installed, validates with `nbformat.validate(nb)`.
- Writes to `output_dir / f"{model['id']}.ipynb"`.

`generate_all(output_dir)`:

- Builds the catalogue via `build_catalog()`.
- Generates one notebook per model.
- Returns the list of generated file paths.

**CLI:** `python3 server/notebook_generator.py --all` or `--single Arima` or `--category deep`.

