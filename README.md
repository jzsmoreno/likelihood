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

The repository includes a dependency-free web catalogue (`server.py`) that
introspects the package sources through `model_catalog.py`. Nothing is hardcoded:
the model list, signatures, parameters, methods, examples and notebooks are all
derived from the real implementation.

Start the server:

```bash
python3 server.py                 # http://127.0.0.1:8000
python3 server.py --port 8080 -v  # custom port, log every request
python3 server.py --list          # print the catalog and exit
```

The browser UI is available at `/` (search, category filters, model cards and a
detail modal) and `/preview/<model_id>` (full documentation plus an interactive
playground). `/docs` renders the OpenAPI document served by `/api/openapi.json`.
Assets live in `models/` and are served with `ETag` and `Cache-Control` headers.

The catalog is re-introspected when a model source changes (`LM_CATALOG_TTL`), so
editing `likelihood/models` is reflected without restarting the server.

### API

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/api/health` | Liveness probe with counters, per endpoint metrics and job stats |
| `GET` | `/api/openapi.json` | OpenAPI 3.1 document generated from the catalog |
| `GET` | `/docs` | Small viewer for the OpenAPI document |
| `GET` | `/api/categories` | Categories with their model counts |
| `GET` | `/api/models` | Model list; accepts `category`, `q`, `tags`, `limit` and `offset` |
| `GET` | `/api/models/<id>` | Everything known about one model (incl. playground spec) |
| `GET` | `/api/models/<id>/source` | Raw source; `?download=1` sends it as `.py`, `?diff=<ref>` a git diff |
| `GET` | `/api/models/<id>/versions` | Commits that touched the source of the model (`?limit=`) |
| `GET` | `/api/search?q=<term>` | Faceted search: results, suggestions and `facets` |
| `GET` | `/api/playground/<id>` | Playground specification of one model |
| `GET` | `/api/playground/jobs` | Most recent asynchronous playground jobs |
| `GET` | `/api/playground/jobs/<job_id>` | State and results of one job |
| `GET` | `/api/models/<id>/invoke` | Methods that may be called on a model (with their arguments) |
| `POST` | `/api/models/<id>/invoke` | `{"method": ..., "arguments": {...}}` builds the model and calls it |
| `POST` | `/api/models/<id>/sample` | `{"method": ..., "parameters": {...}}` returns dummy JSON arguments |
| `POST` | `/api/models/<id>/sessions` | Build the model once and keep it alive for several calls |
| `GET` | `/api/sessions` | Open model sessions |
| `GET` | `/api/sessions/<id>` | State of one session (`calls`, `errors`, `idle_seconds`) |
| `POST` | `/api/sessions/<id>/invoke` | Call a method on the live instance |
| `DELETE` | `/api/sessions/<id>` | Close the session and its worker process |
| `GET` | `/api/notebook/<id>` | Download the generated `.ipynb` (accepts `?parameters=`) |
| `POST` | `/api/playground` | `{"model_id": "...", "parameters": {...}}` builds the model |
| `POST` | `/api/playground/jobs` | One run (or a list of runs) queued in the background |
| `POST` | `/api/notebook` | `{"model_id": "...", "parameters": {...}}` returns the notebook JSON |
| `POST` | `/api/search` | **Deprecated** alias of `GET /api/search` |

Every key of the JSON payloads is `snake_case` (`category_label`,
`short_description`, `parameter_count`, `has_more`, `model_id`, ...), matching the
query parameters (`offset`, `limit`, `tags`). The rename from the previous mixed
casing (`categoryLabel`, `hasMore`, ...) is a breaking change for clients that
consumed the old names.

`GET /api/models` is backwards compatible: without `limit` it returns every
model. With pagination the response adds `matched`, `offset`, `limit`,
`has_more` and `next` (the URL of the following page, also advertised through
`Link: <...>; rel="next"`). `GET /api/models`, `/api/categories`,
`/api/models/<id>` and `/api/openapi.json` also send `ETag` and `Cache-Control`,
so conditional requests answer `304` while the model sources do not change.

`POST /api/search` still works but is deprecated: it answers with
`Deprecation: true`, a `Sunset` date (`LM_SEARCH_ALIAS_SUNSET`, default
`2026-12-31`) and `Link: </api/search?q=...>; rel="successor-version"`. Use
`GET /api/search` instead.

```bash
curl -s http://127.0.0.1:8000/api/models?category=regression&limit=2 | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/search?q=deep\&tags=Deep%20Learning | python3 -m json.tool
curl -s -o Arima.py "http://127.0.0.1:8000/api/models/Arima/source?download=1"
curl -s "http://127.0.0.1:8000/api/models/Arima/source?diff=HEAD~1" | python3 -m json.tool
```

`GET /api/search` returns `facets`: the number of matches per category and per
tag, computed **before** the `tags` filter is applied, so the counters show how
many models each tag would keep. Tags are comma separated and may contain
spaces (`tags=Deep%20Learning,Regression`). `GET /api/models` accepts the same
`tags` filter.

### Playground

Every preview page embeds a playground built from the real `__init__`
signature: adjust the hyperparameters, press **Ejecutar modelo** and the server
constructs the object in a child Python process, returning its type, `repr`,
public attributes, stdout and the elapsed time. Only JSON values are accepted
(no Python code is evaluated), the payload is limited to 1 MiB and the child
process is killed after `LM_PLAYGROUND_TIMEOUT` seconds (default `60`).

The child also lowers its own limits before importing the model
(`resource.setrlimit` for CPU, address space, file descriptors and file size)
and installs an *import guard* that only applies to the **catalog code itself**
(the model module, its package and anything that lives inside the repository):
the catalog cannot pull in `subprocess`, `pty`, `ftplib`... while a third party
library that needs them internally (matplotlib, torch, tensorflow, corner, ...)
keeps working. The list is `LM_PLAYGROUND_BLOCKED_MODULES`; set it to an empty
value to disable the guard, or extend it to be stricter.

> If a run fails with `ImportError: El playground no permite importar '<módulo>'`,
> the *catalog code* (not a library) tried to import a blocked module: customise
> `LM_PLAYGROUND_BLOCKED_MODULES` for that deployment.

```bash
curl -s -X POST http://127.0.0.1:8000/api/playground \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "HMM", "parameters": {"n_states": 3, "n_observations": 4}}' \
  | python3 -m json.tool
```

#### Asynchronous jobs

`POST /api/playground/jobs` accepts the same payload as `POST /api/playground`
(plus a batch: `{"jobs": [{"model_id": ..., "parameters": {...}}, ...]}` or a
bare JSON list) and answers `202` with a `job_id`, a `poll_url` and a `Location`
header. A pool of `LM_JOBS_WORKERS` threads runs the requests while the client
polls `GET /api/playground/jobs/<job_id>`:

```bash
curl -s -X POST http://127.0.0.1:8000/api/playground/jobs \
  -H 'Content-Type: application/json' \
  -d '{"jobs": [{"model_id": "HMM", "parameters": {"n_states": 3, "n_observations": 4}},
                {"model_id": "Arima", "parameters": {}}]}' | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/playground/jobs/<job_id> | python3 -m json.tool
```

`status` is one of `pending`, `running`, `succeeded`, `partial` (some runs
failed) or `failed`; `results` holds one entry per run. Jobs are pruned after
`LM_JOBS_RETENTION` seconds and at most `LM_MAX_JOBS` are kept in memory.

#### Notebooks with the playground values

`POST /api/notebook` (and `GET /api/notebook/<id>?parameters=<json>`) accepts the
same `parameters` object: the validated values become the defaults of the
generated `ipywidgets` controls, so the notebook opens with the experiment you
just ran.

### Invocación de métodos (inferencia)

Además de construir el modelo, la API puede **llamar a sus métodos** — `fit`,
`predict`, `sample`, `forward`, ... — con argumentos JSON validados contra la
firma real. Cada llamada se ejecuta en un proceso hijo desechable con los mismos
límites y el mismo sandbox que el playground.

```bash
# 1) qué métodos se pueden llamar y con qué argumentos
curl -s http://127.0.0.1:8000/api/models/HMM/invoke | python3 -m json.tool

# 2) construir el modelo y llamar a un método (sin estado entre llamadas)
curl -s -X POST http://127.0.0.1:8000/api/models/HMM/invoke \
  -H 'Content-Type: application/json' \
  -d '{"method": "viterbi",
       "parameters": {"n_states": 2, "n_observations": 3},
       "arguments": {"sequence": [0, 1, 2, 1]}}' | python3 -m json.tool
```

La respuesta incluye `result` (proyectado a JSON: listas, dicts, `ndarray` y
`DataFrame` se convierten), `result_type`, `result_repr`, `elapsed`, `stdout` y
`stderr`. Las listas de JSON se transforman en `ndarray`/`DataFrame` cuando la
firma anotada lo indica.

**Datos de ejemplo.** `POST /api/models/<id>/sample` recibe el método y los
parámetros del constructor y devuelve argumentos JSON listos para invocar, con
la forma derivada de esos parámetros (número de estados/observaciones, `window`,
`n_features`...):

```bash
curl -s -X POST http://127.0.0.1:8000/api/models/HMM/sample \
  -H 'Content-Type: application/json' \
  -d '{"method": "viterbi",
       "parameters": {"n_states": 2, "n_observations": 3}}' | python3 -m json.tool
# {"arguments": {"sequence": [0, 1, 0]}, "parameters": {...}, ...}
```

Es lo que usa el botón **🧪 Datos de ejemplo** de la pestaña *Invocar método*:
rellena los argumentos sin escribir JSON a mano, y sigue funcionando cuando los
parámetros del constructor son `np.ndarray`, `DataFrame`, listas o diccionarios.
Los valores propuestos respetan el **default real** del constructor (`n_models=5`,
`num_neurons=128`...) y, cuando el parámetro es obligatorio, usan valores por
nombre (`n_states=2`, `n_observations=3`, `num_classes=3`, `units=8`,
`input_shape_parm=4`, `datapoints` como serie 1-D...).

**Constructores que necesitan Python.** Algunos modelos reciben objetos que no
se pueden expresar como JSON (`AutoQL(env, model)`, `GetInsights(model)`,
`OptionCriticEnv(episodes)` con claves enteras). `GET /api/playground/<id>` y
`GET /api/models/<id>/invoke` lo indican con `"buildable": false` y la lista
`unsupported` (`name`, `type`, `reason`); la API responde `400` explicando qué
parámetro necesita un objeto Python y la web deshabilita los botones en vez de
fallar con un error genérico. El resto de modelos (`HMM`, `Arima`, `GANRegressor`,
`AutoClassifier`, `EnsembleClassifier`, `SimulationEngine`...) se construyen sin
problema con los valores propuestos.

**Modelos Keras (`call` y capas internas).** Cuando la instancia expone `built`
(Keras), la API la construye antes de ejecutar el método: `call` deja de fallar
con `'NoneType' object is not callable` porque `build` crea el
encoder/decoder/classifier con la forma del argumento de entrada. Los argumentos
anotados como `tf.Tensor`/`np.ndarray` se convierten a `ndarray` en el proceso
hijo **tanto en llamadas sin estado como en sesiones**, y los errores de Keras se
devuelven con una explicación que nombra el argumento afectado (por ejemplo,
`'x' (tf.Tensor)`). Si un argumento de tipo tensor/matriz recibe un número o un
texto en lugar de una lista, la API responde `400` indicando el campo antes de
llegar a Keras.

**Política de métodos.** Solo se pueden invocar métodos públicos sin efectos
secundarios: se excluyen propiedades, entradas inventadas por el docstring
(`` `fit ` ``), nombres privados y todo lo que contenga `save`, `load`, `plot`,
`write`, `download`, `delete`, `reset`, `remove`, `close` o `shutdown`. Se puede
cambiar con:

| Variable | Efecto |
| -------- | ------ |
| `LM_INVOKE` | `0` deshabilita por completo `/invoke` y las sesiones |
| `LM_INVOKE_METHODS` | lista blanca explícita (sustituye la política automática) |
| `LM_INVOKE_BLOCKED_METHODS` | nombres a bloquear además de los anteriores |
| `LM_INVOKE_ALLOW_SIDE_EFFECTS` | `1` permite también `save`/`load`/`plot`/... |

Los errores se distinguen: `400` argumentos inválidos (`unknown`, `missing`,
`invalid` en el cuerpo), `403` método bloqueado, `404` modelo o método
inexistente, `422` excepción del modelo (con `detail` = traza), `429` demasiadas
llamadas simultáneas (`LM_INVOKE_MAX_CONCURRENCY`) y `504` timeout.

#### Sesiones con estado

`POST /api/models/<id>/invoke` reconstruye el modelo en cada llamada, así que no
sirve para `fit` → `predict`. Para eso están las sesiones: construyen la
instancia **una vez** en un worker dedicado y aceptan llamadas sucesivas. En la
pestaña *Invocar método* la opción **Instancia persistente** viene marcada: la
primera llamada construye la clase y las siguientes reutilizan la misma
instancia (desmárcala para el modo sin estado).

```bash
# crear la sesión (una vez)
SESSION=$(curl -s -X POST http://127.0.0.1:8000/api/models/FourierRegression/sessions \
  -H 'Content-Type: application/json' -d '{"parameters": {}}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')

# entrenar y predecir sobre la MISMA instancia
curl -s -X POST "http://127.0.0.1:8000/api/sessions/$SESSION/invoke" \
  -H 'Content-Type: application/json' \
  -d '{"method": "fit", "arguments": {"X": [[1], [2], [3]], "y": [2, 4, 6]}}'
curl -s -X POST "http://127.0.0.1:8000/api/sessions/$SESSION/invoke" \
  -H 'Content-Type: application/json' \
  -d '{"method": "predict", "arguments": {"X": [[4]]}}'

# inspeccionar y cerrar
curl -s "http://127.0.0.1:8000/api/sessions/$SESSION"
curl -s -X DELETE "http://127.0.0.1:8000/api/sessions/$SESSION"
```

`GET /api/sessions` lista las sesiones abiertas (modelo, `calls`, `errors`,
`idle_seconds`). Cada sesión es un proceso propio con `RLIMIT`, serializa sus
llamadas y se cierra sola tras `LM_SESSION_TTL` segundos de inactividad; el
store mantiene como máximo `LM_MAX_SESSIONS` (desaloja la menos usada) y una
llamada que excede `LM_SESSION_TIMEOUT` mata el worker. `LM_SESSIONS=0`
deshabilita todo el bloque.

En la pestaña *Invocar método*, **🚀 Desplegar como endpoint** crea la sesión con
los parámetros del constructor y muestra la URL
`POST /api/sessions/<id>/invoke` junto con un `curl` listo para copiar, de modo
que la instancia queda publicada entre llamadas (`fit` → `predict`) hasta que
expire por inactividad.

> **Security note:** the playground imports the package and builds real model
> objects. The server binds to `127.0.0.1` by default; keep it that way unless
> you trust every client that can reach the port. Launch it with
> `--no-playground` (or `LM_PLAYGROUND=0`) to disable execution entirely, or
> with `LM_INVOKE=0` to keep the catalog read-only.

### Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `LM_HOST` | `127.0.0.1` | Interface to bind (`SERVER_HOST` is also accepted) |
| `LM_PORT` | `8000` | TCP port (`SERVER_PORT` is also accepted) |
| `LM_API_TOKEN` | *(empty)* | Require `Authorization: Bearer <token>` on `/api` (health stays open) |
| `LM_CORS_ORIGINS` | *(empty)* | Comma separated allowed origins (`*` allows any) |
| `LM_RATE_LIMIT` | `0` | Requests per minute and client (`0` disables the limiter) |
| `LM_REQUEST_TIMEOUT` | `30` | Seconds a client may take to send its request |
| `LM_CATALOG_TTL` | `2` | Seconds between checks of the model sources (`0` disables hot reload) |
| `LM_API_CACHE_CONTROL` | `public, max-age=60, must-revalidate` | `Cache-Control` of the cacheable JSON endpoints |
| `LM_SEARCH_ALIAS_SUNSET` | `2026-12-31` | `Sunset` date advertised by `POST /api/search` |
| `LM_PLAYGROUND` | `1` | Set to `0`/`false` to disable `POST /api/playground` |
| `LM_PLAYGROUND_TIMEOUT` | `60` | Maximum seconds per playground run |
| `LM_PLAYGROUND_CPU` | `30` | CPU seconds the child process may consume |
| `LM_PLAYGROUND_MEMORY` | `2048` | Address space of the child, in MiB |
| `LM_PLAYGROUND_FILES` | `128` | Open file descriptors of the child |
| `LM_PLAYGROUND_FILE_SIZE` | `32` | Largest file the child may write, in MiB |
| `LM_PLAYGROUND_BLOCKED_MODULES` | `ftplib,pty,smtplib,subprocess,telnetlib,webbrowser,xmlrpc` | Modules the child may not import (empty disables the guard) |
| `LM_PLAYGROUND_MAX_BATCH` | `10` | Maximum number of runs in one job |
| `LM_JOBS_WORKERS` | `2` | Worker threads of the job queue |
| `LM_JOBS_RETENTION` | `900` | Seconds a finished job stays in memory |
| `LM_MAX_JOBS` | `200` | Maximum number of jobs kept in memory |
| `LM_INVOKE` | `1` | Set to `0`/`false` to disable `/invoke` and the sessions |
| `LM_INVOKE_TIMEOUT` | `LM_PLAYGROUND_TIMEOUT` | Maximum seconds per method call |
| `LM_INVOKE_MAX_CONCURRENCY` | `4` | Method calls that may run at the same time |
| `LM_INVOKE_METHODS` | *(empty)* | Explicit allowlist of method names (replaces the policy) |
| `LM_INVOKE_BLOCKED_METHODS` | *(empty)* | Extra method names to block |
| `LM_INVOKE_ALLOW_SIDE_EFFECTS` | `0` | Allow `save`/`load`/`plot`/... methods too |
| `LM_SESSIONS` | `1` | Set to `0`/`false` to disable the stateful sessions |
| `LM_MAX_SESSIONS` | `4` | Live model instances kept in memory |
| `LM_SESSION_TTL` | `900` | Seconds an idle session stays alive |
| `LM_SESSION_TIMEOUT` | `LM_INVOKE_TIMEOUT` | Maximum seconds per call on a session |

The same options are available as flags: `--token`, `--cors-origin` (repeatable),
`--rate-limit`, `--no-playground` and `--no-invoke` (read-only catalog).

### Robustness and observability

* Request bodies are read in 64 KiB chunks up to 1 MiB, both with
  `Content-Length` and `Transfer-Encoding: chunked`; a client that declares more
  bytes than it sends gets a `400`, a stalled one a `408`, an oversized one a
  `413` and a non JSON content type a `415`.
* Errors are mapped to their HTTP status (`400`, `401`, `404`, `405`, `408`,
  `413`, `415`, `422`, `429`, `500`, `503`, `504`) and logged with the endpoint,
  the status, the latency and an `X-Request-Id` that is also returned to the
  client.
* `GET /api/health` reports `metrics.endpoints` with the request count, the
  status distribution and the `p50`/`p95`/`max` latency per endpoint, plus the
  job queue and catalog reload counters.
* The catalog is rebuilt automatically when a model source changes (checked at
  most once every `LM_CATALOG_TTL` seconds), so editing the package no longer
  requires a restart. `--package-dir` keeps working as before.

## Notebook generation

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
python3 -m pytest tests/test_server.py -v   # web server, API and playground
python3 -m pytest tests/test_hmm.py -v      # model tests (requires the full extras)
```

`tests/test_server.py` boots the real HTTP server on a free port, so it covers
routing, headers, caching (static and JSON `ETag`s), JSON projections, template
rendering, error pages, the raw source download, the git diff/versions
endpoints, the sandbox of the playground subprocess, the asynchronous jobs, the
CORS/token/rate-limit layers and the body reader (chunked, truncated and
oversized payloads) without any extra dependency beyond `pytest`.
