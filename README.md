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
playground). Assets live in `models/` and are served with `ETag` and
`Cache-Control` headers.

### API

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/api/health` | Liveness probe with counters and package version |
| `GET` | `/api/categories` | Categories with their model counts |
| `GET` | `/api/models` | Model list; accepts `category`, `q`, `limit` and `offset` |
| `GET` | `/api/models/<id>` | Everything known about one model (incl. playground spec) |
| `GET` | `/api/models/<id>/source` | Raw source of the class; `?download=1` sends it as `.py` |
| `GET` | `/api/search?q=<term>&limit=<n>` | Search results plus autocomplete suggestions |
| `GET` | `/api/playground/<id>` | Playground specification of one model |
| `GET` | `/api/notebook/<id>` | Download the generated `.ipynb` |
| `POST` | `/api/playground` | `{"model_id": "...", "parameters": {...}}` builds the model |
| `POST` | `/api/notebook` | `{"model_id": "..."}` returns the notebook JSON |

`GET /api/models` is backwards compatible: without `limit` it returns every
model. With pagination the response adds `matched`, `offset`, `limit` and
`hasMore`. `POST /api/search` still works but is deprecated (it answers with
`Deprecation: true`); use `GET /api/search` instead.

```bash
curl -s http://127.0.0.1:8000/api/models?category=regression&limit=2 | python3 -m json.tool
curl -s -o Arima.py "http://127.0.0.1:8000/api/models/Arima/source?download=1"
```

### Playground

Every preview page embeds a playground built from the real `__init__`
signature: adjust the hyperparameters, press **Ejecutar modelo** and the server
constructs the object in a child Python process, returning its type, `repr`,
public attributes, stdout and the elapsed time. Only JSON values are accepted
(no Python code is evaluated), the payload is limited to 1 MiB and the child
process is killed after `LM_PLAYGROUND_TIMEOUT` seconds (default `60`).

```bash
curl -s -X POST http://127.0.0.1:8000/api/playground \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "HMM", "parameters": {"n_states": 3, "n_observations": 4}}' \
  | python3 -m json.tool
```

> **Security note:** the playground imports the package and builds real model
> objects. The server binds to `127.0.0.1` by default; keep it that way unless
> you trust every client that can reach the port. Launch it with
> `--no-playground` (or `LM_PLAYGROUND=0`) to disable execution entirely.

### Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `LM_HOST` | `127.0.0.1` | Interface to bind (`SERVER_HOST` is also accepted) |
| `LM_PORT` | `8000` | TCP port (`SERVER_PORT` is also accepted) |
| `LM_PLAYGROUND` | `1` | Set to `0`/`false` to disable `POST /api/playground` |
| `LM_PLAYGROUND_TIMEOUT` | `60` | Maximum seconds per playground run |

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
routing, headers, caching, JSON projections, template rendering, error pages, the
raw-source download and the playground subprocess without any extra dependency
beyond `pytest`.
